"""The whole system in one call: ``DeepSecurePipeline.analyze(path) -> AnalysisResult``.

    validate → frames → audio → visual / audio / lip-sync detectors → fuse
             → security → persist → DetectionReport

* Every stage is timed with ``time.perf_counter()`` (``report.stage_latency_ms``).
* A failure inside a modality stage marks that modality unavailable, adds a warning
  and an ERROR audit event, and the analysis carries on. Only an invalid input
  (``InvalidInputError``) stops the run, after an ``input_rejected`` audit event.
* Models are loaded once in ``__init__``; the Streamlit app caches the pipeline.
* Without a repository nothing is persisted and repeat detection is in memory.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from deepsecure.config.settings import Settings, get_settings
from deepsecure.exceptions import InvalidInputError, ModalityUnavailableError
from deepsecure.fusion.fusion_engine import FusionEngine
from deepsecure.fusion.learned_head import HEAD_FILE, LearnedFusionHead
from deepsecure.models.audio_model import AudioDetector
from deepsecure.models.base import unavailable
from deepsecure.models.lipsync_model import LipSyncDetector
from deepsecure.models.visual_model import VisualDetector
from deepsecure.preprocessing.audio_extractor import AudioExtractor
from deepsecure.preprocessing.frame_extractor import FrameExtractor
from deepsecure.preprocessing.input_validator import InputValidator
from deepsecure.schemas import (
    AudioFeatures,
    DetectionReport,
    FrameBatch,
    LipSyncInput,
    ModalityScore,
)
from deepsecure.security.fingerprint import AttackFingerprinter
from deepsecure.security.security_layer import SecurityLayer
from deepsecure.storage.repository import Repository

logger = logging.getLogger(__name__)
Progress = Callable[[str], None]


@dataclass
class AnalysisResult:
    """The report plus the intermediate data the UI plots (crops, mouth/energy)."""

    report: DetectionReport
    frames: FrameBatch | None = None
    audio: AudioFeatures | None = None
    scores: list[ModalityScore] = field(default_factory=list)


class DeepSecurePipeline:
    def __init__(self, settings: Settings | None = None, repository: Repository | None = None):
        s = self.settings = settings or get_settings()
        self.repository = repository
        self.validator = InputValidator(s)
        self.frame_extractor = FrameExtractor(s)
        self.audio_extractor = AudioExtractor(s)
        self.visual, self.audio_det, self.lipsync = (
            VisualDetector(s),
            AudioDetector(s),
            LipSyncDetector(s),
        )
        for det in (self.visual, self.audio_det, self.lipsync):
            det.load()
        head = LearnedFusionHead()
        head.load(s.models_dir / "fusion" / HEAD_FILE)
        self.fusion = FusionEngine(s, head=head)
        self.security = SecurityLayer(
            s,
            fingerprinter=AttackFingerprinter(store=repository, settings=s)
            if repository is not None
            else None,
        )
        self.security.load()

    @property
    def detectors(self) -> dict[str, object]:
        return {"visual": self.visual, "audio": self.audio_det, "lipsync": self.lipsync}

    @property
    def demo_mode(self) -> bool:
        return not all(d.is_trained for d in self.detectors.values())

    def _audit(self, event: str, message: str, **kw) -> None:
        if self.repository is not None:
            try:
                self.repository.log_event(event, message, **kw)
            except Exception:  # auditing must never break an analysis, but it is logged
                logger.exception("could not write audit event %s", event)

    def analyze(
        self, path: Path | str, original_name: str | None = None, progress: Progress | None = None
    ) -> AnalysisResult:
        path = Path(path)
        name = original_name or path.name
        lat: dict[str, float] = {}
        warnings: list[str] = []
        say = progress or (lambda _msg: None)

        def timed(stage: str, fn: Callable):
            say(stage)
            t0 = time.perf_counter()
            try:
                return fn()
            finally:
                lat[stage] = round((time.perf_counter() - t0) * 1000, 2)

        try:
            validated = timed("validate", lambda: self.validator.validate(path))
        except InvalidInputError as exc:
            self._audit("input_rejected", f"{name}: {exc.reason}", level="WARNING")
            raise
        validated = validated.model_copy(update={"original_name": name})
        self._audit(
            "analysis_started",
            f"analysing {name} ({validated.media_type})",
            payload={"sha256": validated.sha256},
        )

        def guarded(stage: str, fn: Callable, fallback=None):
            try:
                return timed(stage, fn)
            except ModalityUnavailableError as exc:
                warnings.append(f"{stage}: {exc.reason}")
                return fallback
            except Exception as exc:  # one broken modality must not crash the analysis
                logger.exception("stage %s failed", stage)
                warnings.append(f"{stage} failed: {exc}")
                self._audit("stage_failed", f"{stage}: {exc}", level="ERROR")
                return fallback

        frames = audio = None
        if validated.media_type in ("video", "image"):
            frames = guarded("extract_frames", lambda: self.frame_extractor.extract(path))
        if validated.media_type in ("video", "audio"):
            audio = guarded("extract_audio", lambda: self.audio_extractor.extract(path))

        scores = [
            guarded(
                "detect_visual",
                lambda: self.visual.predict(frames),
                unavailable("visual", "visual detector failed"),
            ),
            guarded(
                "detect_audio",
                lambda: self.audio_det.predict(audio),
                unavailable("audio", "audio detector failed"),
            ),
            guarded(
                "detect_lipsync",
                lambda: self.lipsync.predict(LipSyncInput(frames=frames, audio=audio)),
                unavailable("lipsync", "lip-sync detector failed"),
            ),
        ]
        fusion = timed("fuse", lambda: self.fusion.fuse(scores, validated.media_type))
        security = timed("security", lambda: self.security.assess(validated, frames, audio, fusion))
        warnings = fusion.warnings + warnings
        report = DetectionReport(
            input=validated,
            fusion=fusion,
            security=security,
            stage_latency_ms=lat,
            demo_mode=self.demo_mode,
            warnings=warnings,
        )
        if self.repository is not None:
            try:
                timed("persist", lambda: self.repository.save_analysis(report, frames, audio))
            except Exception as exc:
                logger.exception("could not save analysis")
                report.warnings.append(f"result could not be saved: {exc}")
        # pydantic copied ``lat`` when the report was built; bring in "persist" too. (The
        # database row itself can't contain its own save time.)
        report.stage_latency_ms = {**lat, "total": round(sum(lat.values()), 2)}
        self._audit(
            "analysis_completed",
            f"{name}: {security.risk_level} (score {security.risk_score})",
            report_id=report.report_id,
            payload={"stage_latency_ms": report.stage_latency_ms},
        )
        return AnalysisResult(report=report, frames=frames, audio=audio, scores=scores)

    def close(self) -> None:
        self.frame_extractor.close()

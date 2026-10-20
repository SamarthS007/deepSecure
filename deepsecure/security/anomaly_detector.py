"""Layer 5b: is this input *unusual*, whatever the detectors say?

Detectors answer "real or fake?" and can be confidently wrong on inputs unlike their
training data (Phase 4B: the toy audio model called white noise "real" with
confidence 1.00). This layer asks a different question: "does this look like
anything we know?". Two parts:

**1. Isolation Forest on the audio summary vector** (fitted on *real* voices only,
``scripts/fit_anomaly.py``). An Isolation Forest builds many random trees that
split the data at random values; unusual points get isolated in few splits, normal
ones need many. ``score_samples`` is higher for normal points. We store the scores
of the real training voices, so a new file's ``anomaly_score`` is "the share of
real training voices that look more normal than this one" (0.995 = more unusual
than 99.5 % of real voices), measured against *held-out* real voices. A second view,
the most extreme single feature (max |z|), covers outliers the forest misses.
Flag above ``1 - anomaly_quantile / 2`` (each view gets half the false-alarm
budget). Reasons name the features furthest from real speech (z-scores).

**2. Rules** (always on; the fallback when no forest is fitted):

=====================  ==========================================================
code                   fires when
=====================  ==========================================================
face_rarely_visible    video, a face in some but < 30 % of sampled frames
audio_clipping         > 1 % of samples at full scale (distorted, or manipulated)
mostly_silent          > 80 % of audio frames are near-silent
noise_like_audio       spectral flatness > 0.45 (measured: white noise 0.56)
very_short             duration < 2 s (too little evidence)
visual_audio_conflict  |visual - audio| > 0.6, both available
mouth_static           speech audio in a video but the mouth barely moves
=====================  ==========================================================

Every finding has a stable ``code`` (used in attack fingerprints) and a readable
``reason`` (shown to the analyst).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from deepsecure.config.settings import Settings, get_settings
from deepsecure.models.lipsync_model import MOUTH_STATIC_REASON
from deepsecure.preprocessing.audio_extractor import summary_feature_names
from deepsecure.schemas import AudioFeatures, FrameBatch, FusionResult, ValidatedInput

logger = logging.getLogger(__name__)

FOREST_FILE = "anomaly_forest.joblib"


@dataclass
class AnomalyResult:
    flag: bool = False
    score: float = 0.0
    codes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def add(self, code: str, reason: str) -> None:
        self.codes.append(code)
        self.reasons.append(reason)
        self.flag = True


def _max_abs_z(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.abs((x - mean) / std).max(axis=-1)


def fit_forest(
    x_real: np.ndarray, settings: Settings, seed: int = 42, calib_frac: float = 0.2,
    **metadata: object,
) -> dict:  # fmt: skip
    """Fit on real-only summary vectors; return the bundle that :class:`AnomalyDetector` loads.

    The real voices are split: 80 % fit the forest (and the per-feature mean/std), the
    other 20 % *calibrate* it: their scores are what a new real voice is compared with.
    Calibrating on the fitting data would make real voices look more normal than unseen
    ones and inflate false alarms (measured: 3 % instead of the 1 % target).
    """
    if len(x_real) < 50:
        raise ValueError(f"need at least 50 real samples to fit the forest, got {len(x_real)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_real))
    n_cal = max(20, int(len(x_real) * calib_frac))
    fit_x, cal_x = x_real[order[n_cal:]], x_real[order[:n_cal]]
    forest = IsolationForest(n_estimators=300, contamination="auto", random_state=seed)
    forest.fit(fit_x)
    mean, std = fit_x.mean(axis=0), fit_x.std(axis=0) + 1e-9
    return {
        "forest": forest,
        "calib_scores": np.sort(forest.score_samples(cal_x)),
        "calib_max_z": np.sort(_max_abs_z(cal_x, mean, std)),
        "mean": mean,
        "std": std,
        "feature_names": summary_feature_names(settings.n_mfcc),
        "n_mfcc": settings.n_mfcc,
        "n_fit": len(fit_x),
        "n_calib": len(cal_x),
        **metadata,
    }


class AnomalyDetector:
    """Isolation Forest (if fitted) + rules -> :class:`AnomalyResult`."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.bundle: dict | None = None

    @property
    def default_path(self) -> Path:
        return self.settings.models_dir / "security" / FOREST_FILE

    def load(self, path: Path | None = None) -> bool:
        """Load a forest you fitted yourself (joblib = pickle). Returns False if missing."""
        path = path or self.default_path
        if not path.is_file():
            logger.info("No anomaly forest at %s: rule checks only", path)
            self.bundle = None
            return False
        bundle = joblib.load(path)
        if bundle.get("n_mfcc") != self.settings.n_mfcc:
            raise ValueError(f"{path} was fitted with n_mfcc={bundle.get('n_mfcc')}; re-fit it")
        self.bundle = bundle
        return True

    @property
    def flag_cut(self) -> float:
        """Flag above this. Two views each get half the false-alarm budget (Bonferroni),
        so real voices are flagged at about ``anomaly_quantile`` overall, not twice that."""
        return 1.0 - self.settings.anomaly_quantile / 2

    def forest_score(self, vector: np.ndarray) -> tuple[float, list[str]]:
        """(how unusual, in [0, 1], and the most deviating features).

        Two views, each expressed as "the share of held-out real voices that look more
        normal": the Isolation Forest score, and the single most extreme feature
        (largest |z|). The forest picks split features at random, so an outlier along
        only a few of the 170 features is rarely isolated quickly; measured: white noise
        with 16 features beyond 4 std was only at the forest's 92nd percentile. The
        max-|z| view catches exactly that case. The higher of the two is returned.
        """
        b = self.bundle
        x = vector.reshape(1, -1).astype(np.float64)
        raw = float(b["forest"].score_samples(x)[0])  # higher = more normal
        forest_pct = 1.0 - np.searchsorted(b["calib_scores"], raw) / len(b["calib_scores"])
        max_z = float(_max_abs_z(x, b["mean"], b["std"])[0])  # higher = more unusual
        z_pct = np.searchsorted(b["calib_max_z"], max_z) / len(b["calib_max_z"])
        z = (vector - b["mean"]) / b["std"]
        top = np.argsort(-np.abs(z))[: self.settings.anomaly_top_features]
        extreme = [
            f"{b['feature_names'][i]} is {z[i]:+.1f} std from real speech"
            for i in top
            if abs(z[i]) >= self.settings.anomaly_feature_z
        ]
        return float(max(forest_pct, z_pct)), extreme

    def assess(
        self,
        validated: ValidatedInput,
        frames: FrameBatch | None,
        audio: AudioFeatures | None,
        fusion: FusionResult,
    ) -> AnomalyResult:
        s = self.settings
        result = AnomalyResult()

        if audio is not None and self.bundle is not None:
            result.score, extreme = self.forest_score(audio.summary_vector)
            if result.score >= self.flag_cut:
                detail = f"; {', '.join(extreme)}" if extreme else ""
                result.add(
                    "audio_unlike_real_speech",
                    f"voice is more unusual than {result.score:.1%} of held-out real "
                    f"voices{detail}",
                )

        if (
            validated.media_type == "video"
            and frames is not None
            and frames.frames_sampled
            # 0 < ratio: a face that shows up only briefly (a known evasion trick). No face
            # at all is "visual unavailable", already handled by fusion; counting it here
            # too would punish the same fact twice.
            and 0 < frames.face_found_ratio < s.anomaly_min_face_ratio
        ):
            result.add(
                "face_rarely_visible",
                f"a face was found in only {frames.face_found_ratio:.0%} of sampled frames",
            )
        if audio is not None:
            if audio.clipping_ratio > s.anomaly_max_clipping:
                result.add(
                    "audio_clipping", f"{audio.clipping_ratio:.1%} of audio samples are clipped"
                )
            if audio.silence_ratio > s.anomaly_max_silence:
                result.add(
                    "mostly_silent", f"{audio.silence_ratio:.0%} of the audio is near-silent"
                )
            names = summary_feature_names(s.n_mfcc)
            flatness = float(audio.summary_vector[names.index("flatness_mean")])
            if flatness > s.anomaly_max_flatness:
                result.add(
                    "noise_like_audio",
                    f"audio is noise-like (spectral flatness {flatness:.2f}), not speech",
                )
        if validated.media_type != "image" and validated.duration_s < s.anomaly_min_duration_s:
            result.add("very_short", f"only {validated.duration_s:.1f} s long: little evidence")

        by_name = {sc.name: sc for sc in fusion.modality_scores if sc.available}
        if "visual" in by_name and "audio" in by_name:
            gap = abs(by_name["visual"].fake_prob - by_name["audio"].fake_prob)
            if gap > s.anomaly_max_score_gap:
                result.add(
                    "visual_audio_conflict", f"face and voice detectors disagree by {gap:.2f}"
                )
        lipsync = next((sc for sc in fusion.modality_scores if sc.name == "lipsync"), None)
        if lipsync is not None and lipsync.reason == MOUTH_STATIC_REASON and audio is not None:
            result.add("mouth_static", "speech is audible but the mouth barely moves")
        return result

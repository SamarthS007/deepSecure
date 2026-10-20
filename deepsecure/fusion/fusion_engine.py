"""Layer 4: combine the three detector scores into one fake probability.

**Weighted fusion** (the default):

1. Keep only the *available* scores (fusion ignores "couldn't judge").
2. Weight of modality *i* = base weight (visual 0.45, audio 0.35, lip-sync 0.20)
   x its confidence (``fusion_confidence_weighting``), renormalised to sum to 1. A
   zero-confidence score (e.g. the audio detector in demo mode) therefore counts for
   nothing. If every available score has confidence 0, the base weights are used.
3. ``fused_fake_prob`` = weighted mean of the available fake probabilities.

**Confidence** = coverage x agreement x evidence confidence:

* *coverage*: share of the base weight that is available, out of the modalities
  that *apply* to this media type (video: all three; audio file: audio; image:
  visual). A photo isn't penalised for having no voice; a video with its audio
  stripped is (an attacker can remove audio on purpose).
* *agreement*: 1 - weighted std of the probabilities / 0.5 (contradicting
  detectors -> low).
* *evidence confidence*: mean of the scores' own confidences, weighted like the fused
  probability (base x confidence). A detector that is nearly abstaining (p ≈ 0.5, low
  confidence) therefore barely moves the fused probability *or* the confidence. With
  base weights, the undecided lip-sync score on genuine videos (≈ 0.47, confidence ≈ 0.15)
  pulled confidence under 0.5 and turned genuine videos SUSPICIOUS (measured on RAVDESS
  calibration actors, DOCUMENTATION §8.15).

Averaging can hide a strong signal (visual 0.9 + audio 0.1 -> 0.5), so the result
also carries ``max_fake_prob`` / ``max_modality`` and a warning when confident
modalities disagree; the security layer (Phase 6) decides how to act on them.

**Learned fusion** (optional, ``fusion_method = "learned"``): a logistic regression
over per-modality features (see ``learned_head.py``). Falls back to weighted, with
a warning, if no fitted head is loaded.
"""

from __future__ import annotations

import logging

import numpy as np

from deepsecure.config.settings import Settings, get_settings
from deepsecure.fusion.learned_head import LearnedFusionHead
from deepsecure.schemas import FusionResult, MediaType, ModalityName, ModalityScore

logger = logging.getLogger(__name__)

APPLICABLE: dict[str, tuple[ModalityName, ...]] = {
    "video": ("visual", "audio", "lipsync"),
    "audio": ("audio",),
    "image": ("visual",),
}
NO_EVIDENCE = "insufficient evidence: no modality could be analysed"


class FusionEngine:
    """Turns a list of :class:`ModalityScore` into a :class:`FusionResult`."""

    def __init__(self, settings: Settings | None = None, head: LearnedFusionHead | None = None):
        self.settings = settings or get_settings()
        self.head = head

    def fuse(self, scores: list[ModalityScore], media_type: MediaType) -> FusionResult:
        s = self.settings
        base = s.fusion_weights.as_dict()
        avail = [sc for sc in scores if sc.available]
        warnings = [f"{sc.name}: untrained model (demo mode), score not meaningful"
                    for sc in avail if not sc.trained]  # fmt: skip
        if not avail:
            return FusionResult(
                fused_fake_prob=0.5, confidence=0.0, method="weighted", weights_used={},
                modality_scores=scores, warnings=[NO_EVIDENCE],
            )  # fmt: skip

        names = [sc.name for sc in avail]
        p = np.array([sc.fake_prob for sc in avail])
        conf = np.array([sc.confidence for sc in avail])
        b = np.array([base[n] for n in names])
        raw = b * conf if s.fusion_confidence_weighting else b.copy()
        if raw.sum() <= 0:
            raw = b.copy()
        w = raw / raw.sum() if raw.sum() > 0 else np.full(len(b), 1 / len(b))

        method = "weighted"
        fused = float(np.dot(w, p))
        weights_used = {n: float(x) for n, x in zip(names, w, strict=True)}
        if s.fusion_method == "learned":
            if self.head is not None and self.head.model is not None:
                fused = self.head.predict_proba(scores)
                method = "learned"
                weights_used = self.head.modality_weights()
            else:
                warnings.append("learned fusion requested but no fitted head: used weighted")

        applicable = APPLICABLE[media_type]
        coverage = sum(base[n] for n in names if n in applicable) / sum(base[n] for n in applicable)
        mean = float(np.dot(w, p))
        spread = float(np.sqrt(np.dot(w, (p - mean) ** 2)))
        agreement = 1.0 - min(1.0, spread / 0.5)
        evidence_conf = float(np.dot(w, conf))  # same weights as the fused probability
        confidence = min(1.0, coverage) * agreement * evidence_conf

        warnings += self._disagreement(avail)
        i_max = int(np.argmax(p))
        return FusionResult(
            fused_fake_prob=float(np.clip(fused, 0.0, 1.0)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            method=method,
            weights_used=weights_used,
            modality_scores=scores,
            available_modalities=names,
            coverage=float(min(1.0, coverage)),
            agreement=agreement,
            max_fake_prob=float(p[i_max]),
            max_modality=names[i_max],
            trained=all(sc.trained for sc in avail),
            warnings=warnings,
        )

    def _disagreement(self, avail: list[ModalityScore]) -> list[str]:
        """Warn when a confident modality says "fake" and another confident one says "real"."""
        s = self.settings
        sure = [sc for sc in avail if sc.confidence >= s.fusion_disagreement_min_conf]
        fake = [sc for sc in sure if sc.fake_prob > s.risk_thresholds.malicious]
        real = [sc for sc in sure if sc.fake_prob <= s.risk_thresholds.suspicious]
        if fake and real:
            f, r = max(fake, key=lambda x: x.fake_prob), min(real, key=lambda x: x.fake_prob)
            return [f"modalities disagree: {f.name} says fake ({f.fake_prob:.2f}) but "
                    f"{r.name} says real ({r.fake_prob:.2f})"]  # fmt: skip
        return []


MultimodalFusionEngine = FusionEngine  # the name used in the project spec

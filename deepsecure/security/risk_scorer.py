"""Layer 5a: turn the fused fake probability into a risk level, with a decision trail.

Base level from the fused probability p (thresholds from settings):

    p <= 0.40        SAFE
    0.40 < p <= 0.70 SUSPICIOUS
    p > 0.70         MALICIOUS        (a value exactly on a boundary -> the lower level)

Then rules that can only **raise** the level, never lower it (a security system
should fail towards caution: a missed scam usually costs more than a false alarm):

1. no evidence at all                         -> at least SUSPICIOUS
2. confidence < low_confidence_threshold      -> SAFE becomes SUSPICIOUS
3. one confident detector says "fake" (> 0.70) -> at least SUSPICIOUS (averaging may hide it)
4. an anomaly was detected                    -> one level up, but no higher than
                                                 SUSPICIOUS when the fused evidence alone
                                                 says SAFE: "unusual" is not evidence of
                                                 fakery, so MALICIOUS needs fake evidence

Every rule that fires adds a sentence to the decision trail shown in the report.
"""

from __future__ import annotations

from deepsecure.config.settings import Settings, get_settings
from deepsecure.fusion.fusion_engine import NO_EVIDENCE
from deepsecure.schemas import FusionResult, RiskLevel

LEVELS: tuple[RiskLevel, ...] = ("SAFE", "SUSPICIOUS", "MALICIOUS")


def raise_to(current: RiskLevel, minimum: RiskLevel) -> RiskLevel:
    """The higher of two levels."""
    return max(current, minimum, key=LEVELS.index)


def one_level_up(level: RiskLevel) -> RiskLevel:
    return LEVELS[min(LEVELS.index(level) + 1, len(LEVELS) - 1)]


class RiskScorer:
    """Map a :class:`FusionResult` (+ anomaly flag) to a risk level and its reasons."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def base_level(self, p: float) -> RiskLevel:
        t = self.settings.risk_thresholds
        if p <= t.suspicious:
            return "SAFE"
        if p <= t.malicious:
            return "SUSPICIOUS"
        return "MALICIOUS"

    def score(
        self, fusion: FusionResult, anomaly_flag: bool = False
    ) -> tuple[RiskLevel, RiskLevel, list[str]]:
        """Returns (final level, base level, decision trail)."""
        s = self.settings
        p = fusion.fused_fake_prob
        base = self.base_level(p)
        level = base
        trail = [f"fused fake probability {p:.2f} -> {base}"]

        if NO_EVIDENCE in fusion.warnings or not fusion.available_modalities:
            level = raise_to(level, "SUSPICIOUS")
            trail.append(
                f"no detector could analyse this file -> {level} (never SAFE without evidence)"
            )

        if level == "SAFE" and fusion.confidence < s.low_confidence_threshold:
            level = "SUSPICIOUS"
            trail.append(
                f"confidence {fusion.confidence:.2f} < {s.low_confidence_threshold:.2f} "
                f"(coverage {fusion.coverage:.2f}, agreement {fusion.agreement:.2f}) "
                f"-> SUSPICIOUS"
            )

        strong = [
            sc
            for sc in fusion.modality_scores
            if sc.available
            and sc.fake_prob > s.risk_thresholds.malicious
            and sc.confidence >= s.fusion_disagreement_min_conf
        ]
        if strong:  # always shown to the analyst; changes the level only if it was SAFE
            top = max(strong, key=lambda sc: sc.fake_prob)
            new = raise_to(level, "SUSPICIOUS")
            trail.append(
                f"{top.name} alone is confident it's fake ({top.fake_prob:.2f})"
                + (f" -> {new}" if new != level else "")
            )
            level = new

        if anomaly_flag:
            # Without fake evidence (base SAFE), an anomaly and a low-confidence rule must not
            # add up to MALICIOUS: both only say "we can't be sure" (measured: genuine 1.5 s
            # voices went SAFE -> SUSPICIOUS (short) -> MALICIOUS ("very short" anomaly)).
            new = raise_to(level, "SUSPICIOUS") if base == "SAFE" else one_level_up(level)
            trail.append(
                f"anomaly detected -> {new}"
                if new != level
                else "anomaly detected (already at the highest level)"
            )
            level = new

        if fusion.available_modalities and not fusion.trained:
            trail.append("note: at least one score came from an untrained model (demo mode)")
        return level, base, trail

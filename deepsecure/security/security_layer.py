"""Layer 5: the cybersecurity decision on top of the ML verdict.

    FusionResult + inputs --AnomalyDetector--> anomaly flag/codes/reasons
                          --RiskScorer------> risk level (+ decision trail)
                          --AttackFingerprinter--> fingerprint, repeat counts
                          ==> SecurityAssessment

A repeat of the attack fingerprint (>= ``repeat_campaign_threshold`` *different* files
in total) is reported as a "possible repeat attack campaign" only when the file itself is
at least SUSPICIOUS *because of evidence*: coarse fingerprints can collide for unrelated
genuine files, so a repeat alone never raises the level, and files no detector could analyse
(e.g. photos without a face) all share one fingerprint, so they never count as a campaign.
An identical re-upload (same SHA-256) is always reported.
"""

from __future__ import annotations

from deepsecure.config.settings import Settings, get_settings
from deepsecure.schemas import (
    AudioFeatures,
    FrameBatch,
    FusionResult,
    SecurityAssessment,
    ValidatedInput,
)
from deepsecure.security.anomaly_detector import AnomalyDetector
from deepsecure.security.fingerprint import AttackFingerprinter
from deepsecure.security.risk_scorer import RiskScorer


class SecurityLayer:
    def __init__(
        self,
        settings: Settings | None = None,
        anomaly: AnomalyDetector | None = None,
        fingerprinter: AttackFingerprinter | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.anomaly = anomaly or AnomalyDetector(self.settings)
        self.fingerprinter = fingerprinter or AttackFingerprinter(settings=self.settings)
        self.scorer = RiskScorer(self.settings)

    def load(self) -> None:
        self.anomaly.load()

    def assess(
        self,
        validated: ValidatedInput,
        frames: FrameBatch | None,
        audio: AudioFeatures | None,
        fusion: FusionResult,
    ) -> SecurityAssessment:
        found = self.anomaly.assess(validated, frames, audio, fusion)
        level, base, trail = self.scorer.score(fusion, anomaly_flag=found.flag)
        if found.flag:
            trail += [f"anomaly: {r}" for r in found.reasons]

        fingerprint, _ = self.fingerprinter.fingerprint(
            validated, frames, audio, fusion, found.codes
        )
        repeats, seen = self.fingerprinter.record(fingerprint, validated.sha256)
        total = repeats + 1
        evidence = bool(fusion.available_modalities)
        if total >= self.settings.repeat_campaign_threshold and level != "SAFE" and evidence:
            trail.append(
                f"possible repeat attack campaign: {total} different files share this attack "
                f"fingerprint"
            )
        if seen:
            trail.append(f"this exact file (same SHA-256) was analysed {seen} time(s) before")

        return SecurityAssessment(
            risk_level=level,
            risk_score=round(100 * fusion.fused_fake_prob, 1),
            base_level=base,
            decision_trail=trail,
            anomaly_flag=found.flag,
            anomaly_score=found.score,
            anomaly_codes=found.codes,
            anomaly_reasons=found.reasons,
            fingerprint=fingerprint,
            repeat_count=repeats,
            file_seen_count=seen,
        )

"""Layer 5c: recognise repeat attacks.

Two different "have we seen this before?" questions:

* **Same file**: the upload's SHA-256 (computed by the validator) matches an earlier
  one. Catches identical re-uploads, e.g. one deepfake sent to many employees.
* **Same attack pattern**: a scammer re-uses a pipeline (same face-swap model, same
  cloned voice) on *different* files. Their hashes differ completely, but their
  analysis results look alike. The **attack fingerprint** is a SHA-256 over a
  coarse summary of the result:

  - media type and a duration bucket (powers of two: 2-4 s, 4-8 s, ...);
  - every available modality's fake probability rounded to ``fingerprint_quantum`` (0.1);
  - which modalities flagged the file (score above the SUSPICIOUS threshold);
  - the anomaly *codes* (not the free-text reasons, whose numbers vary);
  - a coarse audio statistic (spectral-centroid band, 500 Hz wide) and the
    face-found ratio in quarters.

  Rounding is deliberate: tiny differences between two files from the same pipeline
  must not change the fingerprint. The price is collisions (two unrelated files can
  share one), which is why a repeat only becomes a "possible campaign" when the file
  itself is at least SUSPICIOUS.

The store is in memory here; Phase 7 replaces it with the SQLite table
``attack_fingerprints`` behind the same two methods.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Protocol

from deepsecure.config.settings import Settings, get_settings
from deepsecure.preprocessing.audio_extractor import summary_feature_names
from deepsecure.schemas import AudioFeatures, FrameBatch, FusionResult, ValidatedInput


class FingerprintStore(Protocol):
    def record(self, fingerprint: str, sha256: str) -> tuple[int, int]:
        """Store one analysis. Returns (number of *other* files seen earlier with this
        fingerprint, number of earlier analyses of this exact file)."""
        ...


class InMemoryFingerprintStore:
    """Distinct files per fingerprint and analyses per file hash, for one running process.

    Re-uploading the same file must not look like a campaign, so a fingerprint
    remembers *which* files produced it, not how many times it was produced.
    """

    def __init__(self) -> None:
        self.files_by_fingerprint: dict[str, set[str]] = defaultdict(set)
        self.analyses_by_file: Counter[str] = Counter()

    def record(self, fingerprint: str, sha256: str) -> tuple[int, int]:
        others = len(self.files_by_fingerprint[fingerprint] - {sha256})
        seen = self.analyses_by_file[sha256]
        self.files_by_fingerprint[fingerprint].add(sha256)
        self.analyses_by_file[sha256] += 1
        return others, seen


def _quantise(x: float, step: float) -> float:
    return round(round(x / step) * step, 6)


def fingerprint_material(
    validated: ValidatedInput,
    frames: FrameBatch | None,
    audio: AudioFeatures | None,
    fusion: FusionResult,
    anomaly_codes: list[str],
    settings: Settings,
) -> dict:
    """The coarse, JSON-serialisable summary that gets hashed (kept for explanations)."""
    q = settings.fingerprint_quantum
    scores = {sc.name: _quantise(sc.fake_prob, q) for sc in fusion.modality_scores if sc.available}
    flagged = sorted(
        sc.name
        for sc in fusion.modality_scores
        if sc.available and sc.fake_prob > settings.risk_thresholds.suspicious
    )
    duration = validated.duration_s
    material = {
        "media": validated.media_type,
        "duration_bucket": int(math.log2(duration)) if duration >= 1 else 0,
        "scores": dict(sorted(scores.items())),
        "flagged": flagged,
        "anomalies": sorted(set(anomaly_codes)),
    }
    if audio is not None:
        centroid = float(
            audio.summary_vector[summary_feature_names(settings.n_mfcc).index("centroid_mean")]
        )
        material["centroid_band_hz"] = int(centroid // 500) * 500
    if frames is not None and frames.frames_sampled:
        material["face_ratio_q"] = _quantise(frames.face_found_ratio, 0.25)
    return material


def attack_fingerprint(material: dict) -> str:
    """SHA-256 hex of the canonical JSON (sorted keys, so the same content -> same hash)."""
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class AttackFingerprinter:
    def __init__(self, store: FingerprintStore | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.store = store or InMemoryFingerprintStore()

    def fingerprint(self, validated, frames, audio, fusion, anomaly_codes) -> tuple[str, dict]:
        material = fingerprint_material(
            validated, frames, audio, fusion, anomaly_codes, self.settings
        )
        return attack_fingerprint(material), material

    def record(self, fingerprint: str, sha256: str) -> tuple[int, int]:
        return self.store.record(fingerprint, sha256)

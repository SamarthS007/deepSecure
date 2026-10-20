"""Data contracts between the layers of DeepSecure.

Every layer receives and returns one of these typed objects instead of a loose
``dict``. That gives us three things:

* pydantic checks the values when the object is built (a probability of 1.7
  raises an error immediately instead of silently producing a wrong verdict);
* each layer's input and output are documented in one place;
* in a viva you can say "layer X takes A and returns B" and point at this file.

Flow: ``ValidatedInput`` -> ``FrameBatch`` + ``AudioFeatures`` ->
``ModalityScore`` (x3) -> ``FusionResult`` -> ``SecurityAssessment`` ->
``DetectionReport``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

MediaType = Literal["video", "audio", "image"]
RiskLevel = Literal["SAFE", "SUSPICIOUS", "MALICIOUS"]
ModalityName = Literal["visual", "audio", "lipsync"]


class _ArrayModel(BaseModel):
    """Base for contracts that carry numpy arrays (pydantic can't validate them natively)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _check_ndim(value: np.ndarray, ndim: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(value).__name__}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {value.shape}")
    return value


# --- Layer 2: preprocessing ------------------------------------------------


class ValidatedInput(BaseModel):
    """A file that passed every input check and is safe to analyse."""

    path: Path
    media_type: MediaType
    extension: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)
    duration_s: float = Field(ge=0)  # 0 for still images
    has_audio_track: bool
    width_px: int | None = Field(None, gt=0)  # images only
    height_px: int | None = Field(None, gt=0)  # images only
    original_name: str | None = None  # the uploader's filename (path may be a temp file)


class FrameBatch(_ArrayModel):
    """Face crops and mouth-movement signals extracted from a video (or one image).

    Two different samplings live here, because the two consumers need different things:

    * **Visual stream** (``frame_sample_fps``, default 2/s over the whole clip):
      ``frames_sampled`` frames were examined, a face was found in ``faces_found``
      (= N) of them, giving ``face_crops`` taken at ``crop_timestamps_s``.
    * **Lip-sync stream** (``lipsync_sample_fps``, default 15/s over the first
      ``lipsync_max_s`` seconds): a dense timeline of T points, ``timestamps_s``,
      with ``mouth_open_seq`` and ``mouth_landmarks`` (NaN where no face/landmarks).

    For a still image: ``frames_sampled == 1``, at most one crop, an empty lip-sync
    timeline (T = 0: a photo has no motion) and ``fps == 0``. A **group photo** (no face
    big enough for one full-image pass, found by a tiled re-scan) has ``group_scan`` set
    and one crop per face (N >= 1, ``face_found_ratio`` 1.0); ``crop_source_px`` is the
    side, in original pixels, of the square each crop was cut from (small faces are
    blurrier once enlarged to 299, which the visual detector corrects for).
    """

    face_crops: np.ndarray  # (N, 299, 299, 3) uint8 RGB
    crop_timestamps_s: list[float] = Field(default_factory=list)  # (N,) when each crop was taken
    frames_sampled: int = Field(ge=0)  # visual-stream frames examined
    faces_found: int = Field(ge=0)  # N
    face_found_ratio: float = Field(ge=0.0, le=1.0)  # N / frames_sampled
    timestamps_s: list[float]  # (T,) lip-sync timeline, seconds from the start
    mouth_open_seq: np.ndarray  # (T,) mouth opening / face height; NaN = no face
    mouth_landmarks: np.ndarray  # (T, K, 2) lip landmarks in face-crop coords [0, 1]; NaN = none
    fps: float = Field(ge=0.0)  # the video's native frame rate
    group_scan: bool = False  # still image analysed face by face (tiled re-scan)
    crop_source_px: list[float] = Field(default_factory=list)  # (N,) or empty for videos

    @field_validator("face_crops")
    @classmethod
    def _crops_shape(cls, v: np.ndarray) -> np.ndarray:
        _check_ndim(v, 4, "face_crops")
        if v.shape[-1] != 3:
            raise ValueError(f"face_crops must be RGB (last dim 3), got shape {v.shape}")
        return v

    @field_validator("mouth_open_seq")
    @classmethod
    def _mouth_seq_shape(cls, v: np.ndarray) -> np.ndarray:
        return _check_ndim(v, 1, "mouth_open_seq")

    @field_validator("mouth_landmarks")
    @classmethod
    def _landmarks_shape(cls, v: np.ndarray) -> np.ndarray:
        _check_ndim(v, 3, "mouth_landmarks")
        if v.shape[-1] != 2:
            raise ValueError(f"mouth_landmarks last dim must be 2 (x, y), got {v.shape}")
        return v


class AudioFeatures(_ArrayModel):
    """The audio track as a mono 16 kHz waveform plus the features computed from it."""

    waveform: np.ndarray  # (S,) float32 mono
    sample_rate: int = 16_000
    mfcc: np.ndarray  # (n_mfcc, F)
    log_mel: np.ndarray  # (n_mels, F)
    energy_envelope: np.ndarray  # (F,) RMS per hop; used by lip-sync
    summary_vector: np.ndarray  # (D,) pooled statistics for classic ML + anomaly detection
    duration_s: float = Field(ge=0.0)  # length of the full (untrimmed) track
    offset_s: float = Field(0.0, ge=0.0)  # where the trimmed audio starts in the original
    hop_length: int = Field(160, gt=0)  # samples between feature frames
    clipping_ratio: float = Field(0.0, ge=0.0, le=1.0)  # share of samples at full scale
    silence_ratio: float = Field(0.0, ge=0.0, le=1.0)  # share of frames that are near-silent
    # Untrimmed, un-normalised mono signal at sample_rate: what AASIST was trained on.
    raw_waveform: np.ndarray | None = None

    @field_validator("waveform", "energy_envelope", "summary_vector")
    @classmethod
    def _one_dim(cls, v: np.ndarray, info) -> np.ndarray:  # noqa: ANN001 - pydantic passes ValidationInfo
        return _check_ndim(v, 1, info.field_name)

    @field_validator("mfcc", "log_mel")
    @classmethod
    def _two_dim(cls, v: np.ndarray, info) -> np.ndarray:  # noqa: ANN001
        return _check_ndim(v, 2, info.field_name)


# --- Layer 3: AI analysis --------------------------------------------------


class LipSyncInput(_ArrayModel):
    """Lip-sync detector input: the mouth timeline and the audio (either may be missing)."""

    frames: FrameBatch | None = None
    audio: AudioFeatures | None = None
    audio_shift_s: float = 0.0  # move the audio later by this much (makes training negatives)


class ModalityScore(BaseModel):
    """One detector's verdict on one file.

    If the detector couldn't run (e.g. no face in the video), ``available`` is
    False, ``reason`` says why, and the fusion layer ignores this score.
    """

    name: ModalityName
    fake_prob: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    available: bool = True
    reason: str | None = None
    latency_ms: float = Field(0.0, ge=0.0)
    trained: bool = False  # False = demo mode / uncalibrated: the score is not validated
    details: dict[str, float] = Field(default_factory=dict)  # numbers that explain the score


# --- Layer 4: fusion -------------------------------------------------------


class FusionResult(BaseModel):
    """The combined fake probability across the available modalities.

    ``confidence = coverage x agreement x evidence_confidence`` (see fusion_engine.py).
    """

    fused_fake_prob: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    method: Literal["weighted", "learned"]
    weights_used: dict[str, float]  # final weight per available modality (sums to 1)
    modality_scores: list[ModalityScore]
    available_modalities: list[ModalityName] = Field(default_factory=list)
    coverage: float = Field(0.0, ge=0.0, le=1.0)  # share of the applicable evidence we have
    agreement: float = Field(0.0, ge=0.0, le=1.0)  # 1 = modalities agree, 0 = opposite
    max_fake_prob: float = Field(0.5, ge=0.0, le=1.0)  # strongest single "fake" signal
    max_modality: ModalityName | None = None
    trained: bool = False  # every score used came from a trained/calibrated model
    warnings: list[str] = Field(default_factory=list)


# --- Layer 5: cybersecurity ------------------------------------------------


class SecurityAssessment(BaseModel):
    """The security decision built on top of the ML verdict.

    ``risk_score`` is the fused fake probability x 100 (the evidence); ``risk_level`` is
    the decision, which rules may raise (never lower). ``decision_trail`` lists every
    step that set or changed the level, so an analyst can see *why*.
    """

    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=100.0)
    base_level: RiskLevel = "SAFE"  # from the fused probability alone, before any rule
    decision_trail: list[str] = Field(default_factory=list)
    anomaly_flag: bool = False
    anomaly_score: float = Field(0.0, ge=0.0, le=1.0)  # share of real training voices more normal
    anomaly_codes: list[str] = Field(default_factory=list)  # stable ids, used in fingerprints
    anomaly_reasons: list[str] = Field(default_factory=list)  # human-readable explanations
    fingerprint: str = ""
    repeat_count: int = Field(0, ge=0)  # earlier analyses with the same attack fingerprint
    file_seen_count: int = Field(0, ge=0)  # earlier analyses of this exact file (sha256)


# --- Layer 6: output -------------------------------------------------------


class DetectionReport(BaseModel):
    """Everything known about one analysis; this is what the UI and HTML report show."""

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    input: ValidatedInput
    fusion: FusionResult
    security: SecurityAssessment
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    demo_mode: bool = True
    warnings: list[str] = Field(default_factory=list)

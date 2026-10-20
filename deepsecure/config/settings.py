"""Central configuration for DeepSecure.

Every tunable number (thresholds, weights, sample rates, paths) lives here so
that no other module hard-codes a value. Settings are read in this order, later
sources winning:

1. the defaults written below,
2. a ``.env`` file in the repo root,
3. real environment variables prefixed with ``DEEPSECURE_``.

Nested values use a double underscore, e.g.
``DEEPSECURE_RISK_THRESHOLDS__MALICIOUS=0.8``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# deepsecure/config/settings.py -> parents[2] is the repo root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class FusionWeights(BaseModel):
    """How much each modality counts in the weighted-average fusion.

    They don't have to sum to 1: the fusion engine renormalises over the
    modalities that are actually available for a given file.
    """

    visual: float = Field(0.45, ge=0.0)
    audio: float = Field(0.35, ge=0.0)
    lipsync: float = Field(0.20, ge=0.0)

    def as_dict(self) -> dict[str, float]:
        """Return the weights keyed by modality name."""
        return self.model_dump()


class RiskThresholds(BaseModel):
    """Cut-offs on the fused fake probability.

    ``p <= suspicious`` -> SAFE, ``suspicious < p <= malicious`` -> SUSPICIOUS,
    ``p > malicious`` -> MALICIOUS. A value exactly on a boundary falls into
    the lower category.
    """

    suspicious: float = Field(0.40, gt=0.0, lt=1.0)
    malicious: float = Field(0.70, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> RiskThresholds:
        if self.suspicious >= self.malicious:
            raise ValueError("risk_thresholds.suspicious must be lower than malicious")
        return self


class Settings(BaseSettings):
    """All DeepSecure settings. Get the shared instance with :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="DEEPSECURE_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths -------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    logs_dir: Path = PROJECT_ROOT / "logs"
    db_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'deepsecure.db'}"

    # --- Runtime -----------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    # --- Video preprocessing -------------------------------------------------
    frame_sample_fps: float = Field(2.0, gt=0)
    max_frames: int = Field(64, gt=0)
    face_size: int = Field(299, gt=0)
    face_box_padding: float = Field(0.20, ge=0.0)
    # Lip-sync needs a much denser mouth timeline than the visual detector: speech has
    # ~3-6 syllables/s, so by Nyquist we need > 12 samples/s (see DOCUMENTATION §6.15).
    lipsync_sample_fps: float = Field(15.0, gt=0)
    lipsync_max_s: float = Field(10.0, gt=0)  # analyse lip-sync over the first N seconds
    # Group photos: if a still image has no face large enough for one full-image pass, it is
    # re-scanned in overlapping tiles (BlazeFace short-range sees faces >= ~14 % of its input).
    group_tile_px: int = Field(640, gt=0)  # tile side; tiles overlap by half
    group_max_faces: int = Field(20, gt=0)  # most confident faces kept
    group_min_face_px: int = Field(48, gt=0)  # smaller detections are too blurry to judge
    group_detection_confidence: float = Field(0.6, gt=0.0, lt=1.0)  # stricter: many tiles

    # --- Visual detector (Phase 4A) -------------------------------------------
    visual_arch: str = "legacy_xception"  # timm name; the original Xception in timm 1.x
    visual_fallback_arch: str = "resnet50"  # used if visual_arch can't be built
    visual_top_k: int = Field(5, gt=0)  # video score = mean of the k most suspicious crops
    visual_batch_size: int = Field(16, gt=0)  # crops per forward pass (inference and training)
    visual_epochs: int = Field(8, gt=0)
    visual_lr: float = Field(1e-4, gt=0)
    visual_weight_decay: float = Field(1e-4, ge=0)
    visual_freeze_epochs: int = Field(1, ge=0)  # train only the new head for this many epochs
    visual_patience: int = Field(2, ge=1)  # early stopping: epochs without val improvement
    visual_crops_per_video: int = Field(16, gt=0)  # face crops saved per training video
    visual_prep_fps: float = Field(1.0, gt=0)  # sampling rate when preparing training crops

    # --- Audio detector (Phase 4B) ----------------------------------------------
    audio_logreg_c: float = Field(1.0, gt=0)  # inverse regularisation strength
    audio_mlp_hidden: tuple[int, ...] = (256, 64)
    audio_mlp_alpha: float = Field(1e-3, ge=0)  # L2 penalty
    audio_full_confidence_s: float = Field(3.0, gt=0)  # less voiced audio -> lower confidence
    # "aasist": pretrained raw-waveform network (models/audio/aasist/AASIST.pth);
    # "classic": our MFCC-statistics classifier (models/audio/audio_classifier.joblib).
    # If the chosen model is missing, the other is used; if both are, demo mode.
    audio_backend: Literal["aasist", "classic"] = "aasist"
    # Which signal AASIST sees: "raw" = untrimmed (as trained) or "trimmed" (silence removed;
    # immune to the ASVspoof silence shortcut). Chosen from measurements, DOCUMENTATION §7.4.
    aasist_input: Literal["raw", "trimmed"] = "raw"
    # "auto": our fine-tuned AASIST (In-the-Wild + ASVspoof) if present, else the published one
    aasist_variant: Literal["auto", "pretrained", "finetuned"] = "auto"
    aasist_max_windows: int = Field(8, gt=0)  # 4 s windows, 50 % overlap -> ~18 s analysed
    aasist_threshold: float = Field(0.5, gt=0.0, lt=1.0)  # replaced by the measured dev EER point
    # Voiced seconds for full confidence with AASIST. Its measured dev EER (0.4 %) is on short
    # ASVspoof utterances (genuine dev: 10th percentile 1.13 s, median 1.76 s voiced), so the
    # 3 s used for the MFCC classifier would halve its confidence on its own home ground.
    aasist_full_confidence_s: float = Field(1.0, gt=0)

    # --- Lip-sync baseline (Phase 4C) -----------------------------------------
    lipsync_grid_hz: float = Field(25.0, gt=0)  # even grid both signals are resampled onto
    lipsync_max_lag_s: float = Field(0.5, gt=0)  # search audio/video offsets in +-0.5 s
    lipsync_sync_tolerance_s: float = Field(0.1, ge=0)  # offsets this small count as "in sync"
    lipsync_min_valid_s: float = Field(1.0, gt=0)  # need this much face + audio overlap
    lipsync_full_confidence_s: float = Field(3.0, gt=0)  # less overlap -> confidence scaled down
    # A still face jitters ~0.0004 (measured); below this the mouth isn't really moving.
    lipsync_min_mouth_std: float = Field(0.002, ge=0)
    # Uncalibrated fallback (hand-set, not fitted to data; see lipsync_model.py):
    # fake_prob = sigmoid(slope * ((midpoint - sync_corr) + (peak_corr - sync_corr))).
    lipsync_corr_midpoint: float = Field(0.3, ge=-1.0, le=1.0)
    lipsync_corr_slope: float = Field(8.0, gt=0)

    # --- Audio preprocessing -----------------------------------------------
    audio_sr: int = Field(16_000, gt=0)
    n_mfcc: int = Field(40, gt=0)
    n_mels: int = Field(64, gt=0)
    hop_length: int = Field(160, gt=0)  # 160 samples at 16 kHz = 10 ms
    n_fft: int = Field(400, gt=0)  # 400 samples at 16 kHz = 25 ms analysis window
    audio_trim_top_db: float = Field(30.0, gt=0)  # quieter than peak-30 dB counts as silence
    min_voiced_audio_s: float = Field(0.25, gt=0)  # need this much non-silent audio

    # --- Fusion and security -------------------------------------------------
    fusion_weights: FusionWeights = FusionWeights()
    fusion_method: Literal["weighted", "learned"] = "weighted"  # learned needs a fitted head
    fusion_confidence_weighting: bool = True  # scale each weight by that score's confidence
    # Disagreement warnings only consider scores at least this confident.
    fusion_disagreement_min_conf: float = Field(0.3, ge=0.0, le=1.0)
    risk_thresholds: RiskThresholds = RiskThresholds()
    low_confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)

    # --- Cybersecurity layer (Phase 6) ----------------------------------------------
    # Isolation Forest: flag audio more unusual than (1 - quantile) of real training voices.
    anomaly_quantile: float = Field(0.01, gt=0.0, lt=0.5)
    anomaly_top_features: int = Field(3, ge=1)  # most deviating features named in reasons
    anomaly_feature_z: float = Field(4.0, gt=0)  # |z| above this counts as "extreme"
    anomaly_min_face_ratio: float = Field(0.3, ge=0.0, le=1.0)
    anomaly_max_clipping: float = Field(0.01, ge=0.0, le=1.0)
    anomaly_max_silence: float = Field(0.8, ge=0.0, le=1.0)
    anomaly_min_duration_s: float = Field(2.0, ge=0.0)  # shorter = too little evidence
    # Measured: white noise 0.56 vs speech-like test signals <= 0.29 (real speech: TBD).
    anomaly_max_flatness: float = Field(0.45, ge=0.0, le=1.0)
    anomaly_max_score_gap: float = Field(0.6, ge=0.0, le=1.0)  # |visual - audio|
    fingerprint_quantum: float = Field(0.1, gt=0.0, le=1.0)  # scores rounded to this step
    repeat_campaign_threshold: int = Field(2, ge=2)  # same fingerprint this often -> campaign

    # --- Upload limits -----------------------------------------------------
    max_upload_mb: float = Field(200, gt=0)
    min_duration_s: float = Field(1.0, ge=0)
    max_duration_s: float = Field(600, gt=0)
    video_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    audio_extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    # Images have no duration; these bound their size instead. The pixel limit is checked
    # from the header *before* decoding, which stops "decompression bomb" files.
    max_image_megapixels: float = Field(50.0, gt=0)
    min_image_side_px: int = Field(64, gt=0)  # smaller than this can't hold a usable face

    # --- Dataset splits ----------------------------------------------------
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)  # train / val / test
    split_seed: int = 42

    @property
    def mediapipe_dir(self) -> Path:
        """Folder holding the MediaPipe ``.tflite`` / ``.task`` model files."""
        return self.models_dir / "mediapipe"

    @property
    def samples_dir(self) -> Path:
        """Folder holding the small demo clips used by the UI."""
        return self.data_dir / "samples"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings object (built once, then cached).

    Tests that need different values should construct ``Settings(...)``
    directly instead of mutating this shared instance.
    """
    return Settings()

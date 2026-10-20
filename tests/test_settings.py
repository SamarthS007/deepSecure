"""Settings defaults must match the project specification exactly."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepsecure.config.settings import PROJECT_ROOT, RiskThresholds, Settings


def test_defaults_match_spec() -> None:
    s = Settings(_env_file=None)
    assert s.frame_sample_fps == 2
    assert s.max_frames == 64
    assert s.face_size == 299
    assert s.audio_sr == 16_000
    assert s.n_mfcc == 40
    assert s.n_mels == 64
    assert s.hop_length == 160
    assert s.fusion_weights.as_dict() == {"visual": 0.45, "audio": 0.35, "lipsync": 0.20}
    assert (s.risk_thresholds.suspicious, s.risk_thresholds.malicious) == (0.40, 0.70)
    assert s.max_upload_mb == 200
    assert s.min_duration_s == 1.0
    assert s.max_duration_s == 600


def test_paths_are_inside_repo() -> None:
    s = Settings(_env_file=None)
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert s.mediapipe_dir == PROJECT_ROOT / "models" / "mediapipe"
    assert s.db_url.endswith("data/deepsecure.db")


def test_env_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSECURE_MAX_FRAMES", "16")
    monkeypatch.setenv("DEEPSECURE_RISK_THRESHOLDS__MALICIOUS", "0.8")
    s = Settings(_env_file=None)
    assert s.max_frames == 16
    assert s.risk_thresholds.malicious == 0.8
    assert s.risk_thresholds.suspicious == 0.40  # untouched nested value keeps its default


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        RiskThresholds(suspicious=0.8, malicious=0.5)


def test_phase2_split_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=None)
    assert s.split_ratios == (0.70, 0.15, 0.15)
    assert s.split_seed == 42
    assert ".mp4" in s.video_extensions and ".flac" in s.audio_extensions
    monkeypatch.setenv("DEEPSECURE_SPLIT_RATIOS", "[0.8, 0.1, 0.1]")  # tuples: JSON list
    assert Settings(_env_file=None).split_ratios == (0.8, 0.1, 0.1)

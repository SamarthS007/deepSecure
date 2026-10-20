"""Shared pytest fixtures.

All media is synthesised into pytest's ``tmp_path`` so the suite needs no
dataset, no internet and no GPU.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deepsecure.preprocessing.mediapipe_assets import assets_status
from deepsecure.utils.synthetic_media import (
    mux_audio_into_video,
    write_corrupted_file,
    write_shapes_video,
    write_talking_face_clip,
    write_talking_face_video,
    write_tone_wav,
)

ROOT = Path(__file__).resolve().parents[1]


def isolated_models_dir(tmp_path: Path) -> Path:
    """An empty models dir (no trained checkpoints) that still has the MediaPipe files."""
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    (models / "mediapipe").symlink_to(ROOT / "models" / "mediapipe", target_is_directory=True)
    return models


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
needs_mediapipe_assets = pytest.mark.skipif(
    not all(assets_status().values()), reason="run: uv run python scripts/download_assets.py"
)


@pytest.fixture
def tone_wav(tmp_path: Path) -> Path:
    """A 2-second speech-like tone at 16 kHz."""
    return write_tone_wav(tmp_path / "tone.wav", duration_s=2.0)


@pytest.fixture
def short_wav(tmp_path: Path) -> Path:
    """A 0.5-second clip: below the 1-second minimum duration."""
    return write_tone_wav(tmp_path / "short.wav", duration_s=0.5)


@pytest.fixture
def silent_video(tmp_path: Path) -> Path:
    """A 2-second MP4 with no audio track and no face."""
    return write_shapes_video(tmp_path / "silent.mp4", duration_s=2.0)


@pytest.fixture
def video_with_audio(tmp_path: Path, silent_video: Path, tone_wav: Path) -> Path:
    """A 2-second MP4 with an AAC audio track (skipped if ffmpeg is missing)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    return mux_audio_into_video(silent_video, tone_wav, tmp_path / "with_audio.mp4")


@pytest.fixture
def corrupted_mp4(tmp_path: Path) -> Path:
    """Random bytes saved with an .mp4 extension."""
    return write_corrupted_file(tmp_path / "corrupted.mp4")


@pytest.fixture
def face_video(tmp_path: Path) -> Path:
    """A 2-second silent MP4 of the bundled public-domain face (mouth in motion)."""
    return write_talking_face_video(tmp_path / "face.mp4", duration_s=2.0)


@pytest.fixture
def talking_face_sync(tmp_path: Path) -> Path:
    """3-second face + tone MP4 with the mouth exactly in sync with the audio."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    return write_talking_face_clip(tmp_path / "sync.mp4", duration_s=3.0, lag_s=0.0)


@pytest.fixture
def talking_face_lag(tmp_path: Path) -> Path:
    """Same as ``talking_face_sync`` but the mouth moves 0.5 s after the sound."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    return write_talking_face_clip(tmp_path / "lag.mp4", duration_s=3.0, lag_s=0.5)


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """2 seconds of digital silence."""
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(32_000, dtype=np.float32), 16_000)
    return path


@pytest.fixture
def stereo_44k_wav(tmp_path: Path) -> Path:
    """A 2-second stereo 44.1 kHz WAV (must be converted to mono 16 kHz)."""
    t = np.arange(88_200) / 44_100
    tone = 0.4 * np.sin(2 * np.pi * 330 * t)
    path = tmp_path / "stereo44k.wav"
    sf.write(path, np.stack([tone, 0.5 * tone], axis=1).astype(np.float32), 44_100)
    return path


@pytest.fixture
def delayed_tone_wav(tmp_path: Path) -> Path:
    """0.5 s of silence followed by 1.5 s of tone (tests trimming + offset)."""
    sr = 16_000
    t = np.arange(int(1.5 * sr)) / sr
    y = np.concatenate([np.zeros(sr // 2), 0.5 * np.sin(2 * np.pi * 220 * t)])
    path = tmp_path / "delayed.wav"
    sf.write(path, y.astype(np.float32), sr)
    return path

"""AudioExtractor: loading, normalising, trimming and feature shapes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from deepsecure.config.settings import Settings
from deepsecure.exceptions import InvalidInputError, ModalityUnavailableError
from deepsecure.preprocessing.audio_extractor import AudioExtractor, summary_feature_names
from tests.conftest import needs_ffmpeg


@pytest.fixture
def extractor() -> AudioExtractor:
    return AudioExtractor(Settings(_env_file=None))


def test_feature_shapes(extractor: AudioExtractor, tone_wav: Path) -> None:
    a = extractor.extract(tone_wav)
    frames = a.mfcc.shape[1]
    assert a.sample_rate == 16_000
    assert a.mfcc.shape == (40, frames)
    assert a.log_mel.shape == (64, frames)
    assert a.energy_envelope.shape == (frames,)
    assert frames == pytest.approx(len(a.waveform) / 160, abs=2)  # one frame per 10 ms hop
    assert a.summary_vector.shape == (len(summary_feature_names(40)),) == (170,)
    assert np.isfinite(a.summary_vector).all()
    assert a.waveform.dtype == np.float32
    assert np.max(np.abs(a.waveform)) == pytest.approx(1.0, abs=1e-3)  # peak-normalised


def test_resamples_and_downmixes(extractor: AudioExtractor, stereo_44k_wav: Path) -> None:
    a = extractor.extract(stereo_44k_wav)
    assert a.waveform.ndim == 1
    assert a.duration_s == pytest.approx(2.0, abs=0.01)
    assert len(a.waveform) == pytest.approx(32_000, abs=200)  # 2 s at 16 kHz


def test_trims_leading_silence_and_records_offset(
    extractor: AudioExtractor, delayed_tone_wav: Path
) -> None:
    a = extractor.extract(delayed_tone_wav)
    assert a.offset_s == pytest.approx(0.5, abs=0.05)
    assert len(a.waveform) / 16_000 == pytest.approx(1.5, abs=0.05)
    assert a.duration_s == pytest.approx(2.0, abs=0.01)  # untrimmed length is kept
    assert a.silence_ratio == pytest.approx(0.25, abs=0.05)


def test_silent_audio_is_unavailable(extractor: AudioExtractor, silent_wav: Path) -> None:
    with pytest.raises(ModalityUnavailableError, match="silent"):
        extractor.extract(silent_wav)


def test_clipping_is_measured(extractor: AudioExtractor) -> None:
    t = np.arange(16_000) / 16_000
    clipped = np.clip(3 * np.sin(2 * np.pi * 200 * t), -1, 1).astype(np.float32)
    a = extractor.features_from_waveform(clipped, 16_000)
    assert a.clipping_ratio > 0.5


def test_video_without_audio_is_unavailable(extractor: AudioExtractor, silent_video: Path) -> None:
    with pytest.raises(ModalityUnavailableError, match="no audio track"):
        extractor.extract(silent_video)


@needs_ffmpeg
def test_audio_from_video(extractor: AudioExtractor, video_with_audio: Path) -> None:
    a = extractor.extract(video_with_audio)
    assert a.mfcc.shape[0] == 40
    assert a.duration_s == pytest.approx(2.0, abs=0.1)


@needs_ffmpeg
def test_m4a_goes_through_ffmpeg(extractor: AudioExtractor, tone_wav: Path, tmp_path: Path) -> None:
    m4a = tmp_path / "voice.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tone_wav), "-c:a", "aac", str(m4a)],
        check=True,
    )
    assert extractor.extract(m4a).summary_vector.shape == (170,)


def test_undecodable_audio(extractor: AudioExtractor, tmp_path: Path) -> None:
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(b"not audio at all" * 100)
    with pytest.raises(InvalidInputError):
        extractor.extract(bad)


def test_summary_feature_names_are_unique() -> None:
    names = summary_feature_names(40)
    assert len(names) == len(set(names)) == 170
    assert names[0] == "mfcc0_mean" and names[-1] == "rms_std"

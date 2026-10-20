"""InputValidator: every rejection path and the happy paths."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from deepsecure.config.settings import Settings
from deepsecure.exceptions import InvalidInputError
from deepsecure.preprocessing.input_validator import InputValidator, sha256_of_file
from tests.conftest import needs_ffmpeg


@pytest.fixture
def validator() -> InputValidator:
    return InputValidator(Settings(_env_file=None))


def test_valid_audio(validator: InputValidator, tone_wav: Path) -> None:
    v = validator.validate(tone_wav)
    assert v.media_type == "audio"
    assert v.extension == ".wav"
    assert v.has_audio_track is True
    assert v.duration_s == pytest.approx(2.0, abs=0.05)
    assert v.sha256 == hashlib.sha256(tone_wav.read_bytes()).hexdigest()


def test_valid_video_without_audio(validator: InputValidator, silent_video: Path) -> None:
    v = validator.validate(silent_video)
    assert v.media_type == "video"
    assert v.has_audio_track is False
    assert v.duration_s == pytest.approx(2.0, abs=0.1)


@needs_ffmpeg
def test_valid_video_with_audio(validator: InputValidator, video_with_audio: Path) -> None:
    assert validator.validate(video_with_audio).has_audio_track is True


def test_uppercase_extension_is_accepted(validator: InputValidator, tone_wav: Path) -> None:
    upper = tone_wav.with_name("UPPER.WAV")  # a new name: macOS file names are case-insensitive
    shutil.copy(tone_wav, upper)
    assert validator.validate(upper).extension == ".wav"


def test_missing_file(validator: InputValidator, tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="not found"):
        validator.validate(tmp_path / "nope.mp4")


def test_unsupported_extension(validator: InputValidator, tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(InvalidInputError, match="unsupported file type"):
        validator.validate(f)


def test_empty_file(validator: InputValidator, tmp_path: Path) -> None:
    f = tmp_path / "empty.mp4"
    f.touch()
    with pytest.raises(InvalidInputError, match="empty"):
        validator.validate(f)


def test_corrupted_video(validator: InputValidator, corrupted_mp4: Path) -> None:
    with pytest.raises(InvalidInputError, match="corrupted") as err:
        validator.validate(corrupted_mp4)
    assert str(corrupted_mp4.parent) not in err.value.reason  # no file-system paths leaked


def test_too_short(validator: InputValidator, short_wav: Path) -> None:
    with pytest.raises(InvalidInputError, match="minimum"):
        validator.validate(short_wav)


def test_too_long(tone_wav: Path) -> None:
    strict = InputValidator(Settings(_env_file=None, max_duration_s=1.5))
    with pytest.raises(InvalidInputError, match="maximum"):
        strict.validate(tone_wav)


def test_too_large(tone_wav: Path) -> None:
    tiny = InputValidator(Settings(_env_file=None, max_upload_mb=0.01))  # ~10 KB
    with pytest.raises(InvalidInputError, match="limit"):
        tiny.validate(tone_wav)


def test_audio_disguised_as_video(
    validator: InputValidator, tone_wav: Path, tmp_path: Path
) -> None:
    """A WAV renamed to .mp4 has no video stream, so it must be rejected."""
    fake = tmp_path / "audio_only.mp4"
    shutil.copy(tone_wav, fake)
    with pytest.raises(InvalidInputError):
        validator.validate(fake)


@needs_ffmpeg
def test_m4a_audio(validator: InputValidator, tone_wav: Path, tmp_path: Path) -> None:
    m4a = tmp_path / "voice.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tone_wav), "-c:a", "aac", str(m4a)],
        check=True,
    )
    v = validator.validate(m4a)
    assert v.media_type == "audio" and v.has_audio_track


def test_sha256_streams_large_files(tmp_path: Path) -> None:
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * (3 * 1024 * 1024 + 7))  # spans several 1 MiB chunks
    assert sha256_of_file(f) == hashlib.sha256(f.read_bytes()).hexdigest()

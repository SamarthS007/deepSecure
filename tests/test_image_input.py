"""Still-image input: loading, validation, face extraction and the audio guard."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deepsecure.config.settings import Settings
from deepsecure.exceptions import InvalidInputError, ModalityUnavailableError
from deepsecure.preprocessing.audio_extractor import AudioExtractor
from deepsecure.preprocessing.frame_extractor import FrameExtractor
from deepsecure.preprocessing.image_loader import inspect_image, load_image_rgb
from deepsecure.preprocessing.input_validator import InputValidator
from deepsecure.preprocessing.mediapipe_assets import assets_status
from deepsecure.utils.synthetic_media import (
    face_photo_rgb,
    write_corrupted_file,
    write_face_photo,
    write_no_face_image,
    write_truncated_copy,
)
from tests.conftest import needs_mediapipe_assets

MP = 50.0  # default megapixel limit


@pytest.fixture
def validator() -> InputValidator:
    return InputValidator(Settings(_env_file=None))


@pytest.fixture
def face_jpg(tmp_path: Path) -> Path:
    return write_face_photo(tmp_path / "face.jpg")


# --- loader -----------------------------------------------------------------


@pytest.mark.parametrize("ext", [".jpg", ".png", ".webp", ".bmp"])
def test_every_format_loads_as_rgb(tmp_path: Path, ext: str) -> None:
    rgb = load_image_rgb(write_face_photo(tmp_path / f"face{ext}"), MP)
    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    """A phone-style sideways JPEG must come back upright, matching the original."""
    path = write_face_photo(tmp_path / "phone.jpg", exif_rotated=True)
    with Image.open(path) as raw:
        assert raw.size == (480, 640)  # stored sideways
    rgb = load_image_rgb(path, MP)
    assert rgb.shape == (480, 640, 3)  # upright again
    info = inspect_image(path, MP)
    assert (info.width, info.height) == (640, 480)  # the validator reports the upright size
    diff = np.abs(rgb.astype(int) - face_photo_rgb().astype(int)).mean()
    assert diff < 5  # only JPEG noise left


def test_transparent_png_is_composited_on_white(tmp_path: Path) -> None:
    rgba = np.zeros((100, 100, 4), np.uint8)
    rgba[:50, :, :] = (255, 0, 0, 255)  # top half opaque red, bottom half transparent
    path = tmp_path / "alpha.png"
    Image.fromarray(rgba).save(path)
    rgb = load_image_rgb(path, MP)
    assert tuple(rgb[10, 10]) == (255, 0, 0)
    assert tuple(rgb[90, 90]) == (255, 255, 255)


def test_greyscale_becomes_three_channels(tmp_path: Path) -> None:
    path = tmp_path / "grey.png"
    Image.fromarray(np.full((80, 80), 128, np.uint8)).save(path)
    assert load_image_rgb(path, MP).shape == (80, 80, 3)


def test_pixel_limit_is_checked_before_decoding(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    Image.new("RGB", (2000, 1500)).save(path)  # 3 MP, tiny file (all black)
    with pytest.raises(InvalidInputError, match="limit is 2 MP"):
        inspect_image(path, max_megapixels=2)


# --- validator ----------------------------------------------------------------


def test_valid_image(validator: InputValidator, face_jpg: Path) -> None:
    v = validator.validate(face_jpg)
    assert v.media_type == "image"
    assert (v.width_px, v.height_px) == (640, 480)
    assert v.duration_s == 0.0 and v.has_audio_track is False
    assert len(v.sha256) == 64


def test_uppercase_jpeg_extension(validator: InputValidator, face_jpg: Path) -> None:
    upper = face_jpg.with_name("PHOTO.JPEG")
    shutil.copy(face_jpg, upper)
    assert validator.validate(upper).media_type == "image"


def test_png_named_jpg_is_accepted(validator: InputValidator, tmp_path: Path) -> None:
    """The format is read from the contents; a mislabelled real image is harmless."""
    png = write_face_photo(tmp_path / "x.png")
    renamed = tmp_path / "x_really_png.jpg"
    shutil.copy(png, renamed)
    assert validator.validate(renamed).media_type == "image"


def test_video_renamed_to_jpg_is_rejected(
    validator: InputValidator, silent_video: Path, tmp_path: Path
) -> None:
    fake = tmp_path / "video.jpg"
    shutil.copy(silent_video, fake)
    with pytest.raises(InvalidInputError, match="not a supported image"):
        validator.validate(fake)


def test_random_bytes_jpg_is_rejected(validator: InputValidator, tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError):
        validator.validate(write_corrupted_file(tmp_path / "noise.jpg"))


def test_truncated_jpg_is_rejected(
    validator: InputValidator, face_jpg: Path, tmp_path: Path
) -> None:
    cut = write_truncated_copy(face_jpg, tmp_path / "cut.jpg")
    with pytest.raises(InvalidInputError, match="truncated"):
        validator.validate(cut)


def test_tiny_image_is_rejected(validator: InputValidator, tmp_path: Path) -> None:
    path = tmp_path / "icon.png"
    Image.new("RGB", (32, 32)).save(path)
    with pytest.raises(InvalidInputError, match="at least 64 px"):
        validator.validate(path)


def test_gif_is_not_on_the_whitelist(validator: InputValidator, tmp_path: Path) -> None:
    path = tmp_path / "anim.gif"
    Image.new("RGB", (100, 100)).save(path)
    with pytest.raises(InvalidInputError, match="unsupported file type"):
        validator.validate(path)


# --- extractors ---------------------------------------------------------------


def test_audio_extractor_says_images_have_no_audio(face_jpg: Path) -> None:
    with pytest.raises(ModalityUnavailableError, match="no audio"):
        AudioExtractor(Settings(_env_file=None)).extract(face_jpg)


@pytest.fixture(scope="module")
def extractor() -> Iterator[FrameExtractor]:
    if not all(assets_status().values()):
        pytest.skip("run: uv run python scripts/download_assets.py")
    with FrameExtractor(Settings(_env_file=None)) as fx:
        yield fx


@needs_mediapipe_assets
def test_face_photo_gives_one_crop(extractor: FrameExtractor, face_jpg: Path) -> None:
    batch = extractor.extract(face_jpg)  # dispatches on the extension
    assert batch.face_crops.shape == (1, 299, 299, 3)
    assert batch.frames_sampled == batch.faces_found == 1
    assert batch.face_found_ratio == 1.0
    assert batch.crop_timestamps_s == [0.0]
    assert batch.timestamps_s == [] and batch.mouth_open_seq.shape == (0,)
    assert batch.fps == 0.0


@needs_mediapipe_assets
def test_phone_rotated_photo_gives_an_upright_crop(
    extractor: FrameExtractor, face_jpg: Path, tmp_path: Path
) -> None:
    """BlazeFace finds sideways faces too, so the real risk is a *sideways crop*
    reaching a CNN trained on upright faces. The crop must match the upright photo's."""
    phone = write_face_photo(tmp_path / "phone.jpg", exif_rotated=True)
    upright = extractor.extract_image(face_jpg).face_crops[0].astype(int)
    rotated = extractor.extract_image(phone).face_crops[0].astype(int)
    sideways = np.rot90(upright)  # what we'd get if EXIF were ignored
    assert np.abs(rotated - upright).mean() < 10
    assert np.abs(sideways - upright).mean() > 2 * np.abs(rotated - upright).mean()


@needs_mediapipe_assets
def test_no_face_image(extractor: FrameExtractor, tmp_path: Path) -> None:
    batch = extractor.extract(write_no_face_image(tmp_path / "shapes.png"))
    assert batch.faces_found == 0 and batch.face_found_ratio == 0.0
    assert batch.face_crops.shape == (0, 299, 299, 3)

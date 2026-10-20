"""FrameExtractor: sampling maths, crop geometry, face / no-face videos, lip-sync timeline."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from deepsecure.config.settings import Settings
from deepsecure.exceptions import AssetMissingError, InvalidInputError
from deepsecure.preprocessing.frame_extractor import (
    LIP_LANDMARKS,
    FrameExtractor,
    sample_frame_indices,
    square_crop_box,
)
from deepsecure.preprocessing.mediapipe_assets import assets_status
from deepsecure.utils.synthetic_media import syllable_envelope
from tests.conftest import needs_mediapipe_assets

# --- pure functions (no MediaPipe needed) ------------------------------------


def test_sample_indices_at_2fps_from_25fps() -> None:
    assert sample_frame_indices(100, 25.0, 2.0, 64) == [0, 12, 25, 38, 50, 62, 75, 88]


def test_sample_indices_respect_max_frames() -> None:
    assert len(sample_frame_indices(10_000, 25.0, 2.0, 64)) == 64


def test_sample_indices_never_repeat_when_sampling_faster_than_video() -> None:
    idx = sample_frame_indices(30, 10.0, 15.0, 100)  # asks for more than the video has
    assert idx == list(range(30))


def test_square_crop_box_is_square_and_padded() -> None:
    x0, y0, x1, y1 = square_crop_box(100, 100, 80, 100, 640, 480, padding=0.2)
    assert x1 - x0 == y1 - y0 == 120  # max(80, 100) * 1.2
    assert (x0 + x1) / 2 == pytest.approx(140, abs=1)  # still centred on the face


def test_square_crop_box_is_shifted_inside_the_frame() -> None:
    x0, y0, x1, y1 = square_crop_box(0, 0, 100, 100, 640, 480, padding=0.2)
    assert (x0, y0) == (0, 0) and (x1, y1) == (120, 120)
    x0, y0, x1, y1 = square_crop_box(590, 400, 50, 80, 640, 480, padding=0.2)
    assert x1 <= 640 and y1 <= 480 and x1 - x0 == y1 - y0


def test_lip_landmarks_include_inner_lip_points() -> None:
    assert len(LIP_LANDMARKS) == 40
    assert {13, 14} <= set(LIP_LANDMARKS)


def test_missing_assets_raise_clear_error(tmp_path: Path) -> None:
    with pytest.raises(AssetMissingError, match="download_assets"):
        FrameExtractor(mediapipe_dir=tmp_path)


# --- with MediaPipe ---------------------------------------------------------


@pytest.fixture(scope="module")
def extractor() -> Iterator[FrameExtractor]:
    if not all(assets_status().values()):
        pytest.skip("run: uv run python scripts/download_assets.py")
    with FrameExtractor(Settings(_env_file=None)) as fx:
        yield fx


@needs_mediapipe_assets
def test_no_face_video(extractor: FrameExtractor, silent_video: Path) -> None:
    batch = extractor.extract(silent_video)
    assert batch.frames_sampled == 4  # 2 s at 2 fps
    assert batch.faces_found == 0
    assert batch.face_found_ratio == 0.0
    assert batch.face_crops.shape == (0, 299, 299, 3)
    assert np.isnan(batch.mouth_open_seq).all()
    assert len(batch.timestamps_s) == 30  # 2 s at 15 fps


@needs_mediapipe_assets
def test_face_video_shapes(extractor: FrameExtractor, face_video: Path) -> None:
    batch = extractor.extract(face_video)
    assert batch.faces_found == batch.frames_sampled == 4
    assert batch.face_found_ratio == 1.0
    assert batch.face_crops.shape == (4, 299, 299, 3)
    assert batch.face_crops.dtype == np.uint8
    assert batch.crop_timestamps_s == [0.0, 0.48, 1.0, 1.52]
    t = len(batch.timestamps_s)
    assert batch.mouth_open_seq.shape == (t,)
    assert batch.mouth_landmarks.shape == (t, 40, 2)
    assert np.isfinite(batch.mouth_open_seq).all()
    assert np.nanmin(batch.mouth_landmarks) > 0.0 and np.nanmax(batch.mouth_landmarks) < 1.0


@needs_mediapipe_assets
def test_max_frames_cap(face_video: Path) -> None:
    with FrameExtractor(Settings(_env_file=None, max_frames=2, lipsync_max_s=0.5)) as fx:
        batch = fx.extract(face_video)
    assert batch.frames_sampled == 2
    assert len(batch.timestamps_s) == 7  # 0.5 s x min(15, 25) fps -> 7 points


@needs_mediapipe_assets
def test_mouth_timeline_follows_the_audio_rhythm(
    extractor: FrameExtractor, talking_face_sync: Path, talking_face_lag: Path
) -> None:
    """In sync -> strong positive correlation; 0.5 s late at 3 Hz -> opposite phase."""

    def corr(path: Path) -> float:
        batch = extractor.extract(path)
        envelope = syllable_envelope(np.array(batch.timestamps_s))
        return float(np.corrcoef(batch.mouth_open_seq, envelope)[0, 1])

    assert corr(talking_face_sync) > 0.8
    assert corr(talking_face_lag) < -0.8


@needs_mediapipe_assets
def test_corrupted_video_raises(extractor: FrameExtractor, corrupted_mp4: Path) -> None:
    with pytest.raises(InvalidInputError):
        extractor.extract(corrupted_mp4)


@needs_mediapipe_assets
def test_closed_extractor_refuses_work(face_video: Path) -> None:
    fx = FrameExtractor(Settings(_env_file=None))
    fx.close()
    fx.close()  # closing twice is fine
    with pytest.raises(RuntimeError, match="closed"):
        fx.extract(face_video)

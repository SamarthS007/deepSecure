"""Device selection, logging, MediaPipe assets, synthetic media and the UI shell."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
import soundfile as sf
import torch

from deepsecure.config.device import describe_device, get_device
from deepsecure.config.logging_config import setup_logging
from deepsecure.exceptions import AssetMissingError
from deepsecure.preprocessing.mediapipe_assets import (
    DOWNLOAD_HINT,
    assets_status,
    create_face_detector,
    create_face_landmarker,
    require_asset,
)

ROOT = Path(__file__).resolve().parents[1]


# --- device ----------------------------------------------------------------


def test_auto_device_is_a_torch_device() -> None:
    device = get_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}
    assert describe_device(device)


def test_forced_cpu() -> None:
    assert get_device("cpu") == torch.device("cpu")


def test_unavailable_device_falls_back_to_cpu() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    assert get_device("cuda") == torch.device("cpu")


# --- logging ---------------------------------------------------------------


def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    logger = setup_logging("DEBUG", logs_dir=tmp_path)
    n_handlers = len(logger.handlers)
    setup_logging("DEBUG", logs_dir=tmp_path)
    assert len(logging.getLogger("deepsecure").handlers) == n_handlers


# --- MediaPipe assets (fix #1: Tasks API, not mp.solutions) ---------------


def test_missing_asset_gives_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(AssetMissingError, match="download_assets.py"):
        require_asset("face_detector", mediapipe_dir=tmp_path)
    assert "uv run" in DOWNLOAD_HINT


def test_detector_factory_raises_asset_error_not_mediapipe_error(tmp_path: Path) -> None:
    with pytest.raises(AssetMissingError):
        create_face_landmarker(mediapipe_dir=tmp_path)


@pytest.mark.skipif(not all(assets_status().values()), reason="run scripts/download_assets.py")
def test_tasks_api_detectors_run_on_blank_image() -> None:
    import mediapipe as mp

    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=blank)
    with create_face_detector() as detector:
        assert detector.detect(image).detections == []
    with create_face_landmarker() as landmarker:
        assert landmarker.detect(image).face_landmarks == []


# --- synthetic media --------------------------------------------------------


def test_tone_wav_has_expected_duration(tone_wav: Path) -> None:
    info = sf.info(tone_wav)
    assert info.samplerate == 16_000
    assert info.duration == pytest.approx(2.0, abs=0.01)


def test_silent_video_is_readable(silent_video: Path) -> None:
    cap = cv2.VideoCapture(str(silent_video))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 50
        ok, frame = cap.read()
        assert ok and frame.shape == (240, 320, 3)
    finally:
        cap.release()


def test_muxed_video_has_audio_stream(video_with_audio: Path) -> None:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(video_with_audio),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "audio" in probe.stdout


def test_corrupted_file_is_not_openable_as_video(corrupted_mp4: Path) -> None:
    cap = cv2.VideoCapture(str(corrupted_mp4))
    try:
        assert not cap.isOpened() or not cap.read()[0]
    finally:
        cap.release()

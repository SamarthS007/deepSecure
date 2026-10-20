"""MediaPipe model files and detector factories (Tasks API).

Why this module exists: mediapipe 1.x no longer ships the legacy
``mp.solutions`` API (``hasattr(mediapipe, "solutions")`` is False on the
installed version). The replacement is the **Tasks API**, which loads a model
file from disk instead of bundling it inside the package. So we have to
download those files once (``scripts/download_assets.py``) and point the
detectors at them.

Both factories below raise :class:`AssetMissingError` with a clear "run this
command" message if a file is missing, instead of a cryptic mediapipe crash.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deepsecure.config.settings import get_settings
from deepsecure.exceptions import AssetMissingError

DOWNLOAD_HINT = "uv run python scripts/download_assets.py"


@dataclass(frozen=True)
class MediaPipeAsset:
    """One downloadable MediaPipe model file."""

    key: str
    filename: str
    url: str
    purpose: str


# URLs are from Google's MediaPipe model cards (version-pinned "/1/" path,
# verified to return HTTP 200 on 2026-10-14).
ASSETS: dict[str, MediaPipeAsset] = {
    "face_detector": MediaPipeAsset(
        key="face_detector",
        filename="blaze_face_short_range.tflite",
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        ),
        purpose="BlazeFace short-range: face bounding boxes (faces within ~2 m of the camera)",
    ),
    "face_landmarker": MediaPipeAsset(
        key="face_landmarker",
        filename="face_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        ),
        purpose="Face mesh: 478 3-D landmarks per face (we use the lip points for lip-sync)",
    ),
}


def asset_path(key: str, mediapipe_dir: Path | None = None) -> Path:
    """Where the asset should live on disk (it may not exist yet)."""
    folder = mediapipe_dir or get_settings().mediapipe_dir
    return folder / ASSETS[key].filename


def require_asset(key: str, mediapipe_dir: Path | None = None) -> Path:
    """Return the asset's path, or raise ``AssetMissingError`` with the fix."""
    path = asset_path(key, mediapipe_dir)
    if not path.is_file() or path.stat().st_size == 0:
        raise AssetMissingError(
            f"MediaPipe model file missing: {path}. Download it with: {DOWNLOAD_HINT}"
        )
    return path


def assets_status(mediapipe_dir: Path | None = None) -> dict[str, bool]:
    """Map each asset key to whether its file is present (used by the UI)."""
    return {key: asset_path(key, mediapipe_dir).is_file() for key in ASSETS}


def create_face_detector(min_confidence: float = 0.5, mediapipe_dir: Path | None = None):
    """Build a Tasks API ``FaceDetector`` in IMAGE mode.

    IMAGE mode treats every frame independently, which suits our "sample a few
    frames per second" approach. Remember to call ``.close()`` (or use ``with``).
    """
    from mediapipe.tasks.python import BaseOptions, vision

    model_path = require_asset("face_detector", mediapipe_dir)
    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=min_confidence,
    )
    return vision.FaceDetector.create_from_options(options)


def create_face_landmarker(min_confidence: float = 0.5, mediapipe_dir: Path | None = None):
    """Build a Tasks API ``FaceLandmarker`` (one face, IMAGE mode)."""
    from mediapipe.tasks.python import BaseOptions, vision

    model_path = require_asset("face_landmarker", mediapipe_dir)
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
    )
    return vision.FaceLandmarker.create_from_options(options)

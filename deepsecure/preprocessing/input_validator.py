"""Layer 2a: decide whether an uploaded file is safe and sensible to analyse.

This is the system's front door, so it's deliberately strict. Each check
answers one question, cheapest first:

1. Does the file exist?                 -> "file not found"
2. Is the extension on the whitelist?    -> "unsupported file type"
3. Is it non-empty and within the size limit?
4. SHA-256 of the contents (streamed, so a 200 MB file never sits in memory).
5. Can ffprobe parse it, and does it contain the stream its extension promises?
6. Can we actually *decode* it (OpenCV for video, ffmpeg for audio)?
7. Is the duration within [min_duration_s, max_duration_s]?

Still images skip 5-7 and instead go through :mod:`image_loader`: the format is
sniffed from the contents, the pixel count is checked from the header before
decoding (decompression bombs), the image must decode fully, and its shorter
side must be at least ``min_image_side_px``.

Any failure raises :class:`InvalidInputError` with a human-readable ``reason``.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

import cv2

from deepsecure.config.settings import Settings, get_settings
from deepsecure.exceptions import InvalidInputError
from deepsecure.preprocessing.image_loader import inspect_image
from deepsecure.preprocessing.media_probe import probe
from deepsecure.schemas import MediaType, ValidatedInput

logger = logging.getLogger(__name__)

HASH_CHUNK_BYTES = 1024 * 1024  # read 1 MiB at a time


def sha256_of_file(path: Path) -> str:
    """SHA-256 hex digest computed in chunks, so memory use stays constant."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


class InputValidator:
    """Validate uploads and describe them as a :class:`ValidatedInput`.

    Args:
        settings: Optional settings override (tests pass tighter limits).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def media_type_for(self, path: Path) -> MediaType:
        """``"video"``, ``"audio"`` or ``"image"`` from the extension; raise for anything else."""
        s = self.settings
        ext = path.suffix.lower()
        if ext in s.video_extensions:
            return "video"
        if ext in s.audio_extensions:
            return "audio"
        if ext in s.image_extensions:
            return "image"
        allowed = ", ".join(s.video_extensions + s.audio_extensions + s.image_extensions)
        raise InvalidInputError(f"unsupported file type '{ext or '(none)'}'; allowed: {allowed}")

    def validate(self, path: Path | str) -> ValidatedInput:
        """Run every check and return the validated description of the file."""
        path = Path(path)
        if not path.is_file():
            raise InvalidInputError(f"file not found: {path}")

        media_type = self.media_type_for(path)

        size = path.stat().st_size
        if size == 0:
            raise InvalidInputError("file is empty")
        limit = self.settings.max_upload_mb * 1024 * 1024
        if size > limit:
            mb = size / 1024 / 1024
            raise InvalidInputError(
                f"file is {mb:.1f} MB; the limit is {self.settings.max_upload_mb} MB"
            )

        sha256 = sha256_of_file(path)
        if media_type == "image":
            return self._validate_image(path, size, sha256)
        info = probe(path)

        if media_type == "video":
            if not info.has_video:
                raise InvalidInputError("no video stream found in this video file")
            self._check_video_decodes(path)
        else:
            if not info.has_audio:
                raise InvalidInputError("no audio stream found in this audio file")
            self._check_audio_decodes(path)

        duration = info.duration_s if info.duration_s is not None else self._cv2_duration(path)
        if duration is None:
            raise InvalidInputError("could not determine the duration")
        if duration < self.settings.min_duration_s:
            raise InvalidInputError(
                f"clip is {duration:.2f} s long; the minimum is {self.settings.min_duration_s} s"
            )
        if duration > self.settings.max_duration_s:
            raise InvalidInputError(
                f"clip is {duration:.0f} s long; the maximum is {self.settings.max_duration_s} s"
            )

        validated = ValidatedInput(
            path=path.resolve(),
            media_type=media_type,
            extension=path.suffix.lower(),
            size_bytes=size,
            sha256=sha256,
            duration_s=round(duration, 3),
            has_audio_track=info.has_audio,
        )
        logger.info(
            "Validated %s (%s, %.2f s, audio=%s, sha256=%s...)",
            path.name, media_type, duration, info.has_audio, sha256[:12],
        )  # fmt: skip
        return validated

    def _validate_image(self, path: Path, size: int, sha256: str) -> ValidatedInput:
        """Image branch: content-sniffed format, pixel limit, full decode, minimum size."""
        info = inspect_image(path, self.settings.max_image_megapixels)
        shorter = min(info.width, info.height)
        if shorter < self.settings.min_image_side_px:
            raise InvalidInputError(
                f"image is {info.width}x{info.height} px; the shorter side must be at least "
                f"{self.settings.min_image_side_px} px"
            )
        expected = (".jpg", ".jpeg") if info.format == "JPEG" else (f".{info.format.lower()}",)
        if path.suffix.lower() not in expected:
            # e.g. a PNG saved as .jpg: harmless and common, so only logged.
            logger.warning("%s is really a %s file", path.name, info.format)
        logger.info(
            "Validated %s (image %s, %dx%d, sha256=%s...)",
            path.name, info.format, info.width, info.height, sha256[:12],
        )  # fmt: skip
        return ValidatedInput(
            path=path.resolve(),
            media_type="image",
            extension=path.suffix.lower(),
            size_bytes=size,
            sha256=sha256,
            duration_s=0.0,
            has_audio_track=False,
            width_px=info.width,
            height_px=info.height,
        )

    @staticmethod
    def _check_video_decodes(path: Path) -> None:
        """OpenCV must read at least one frame: the frame extractor relies on OpenCV."""
        cap = cv2.VideoCapture(str(path))
        try:
            ok = cap.isOpened() and cap.read()[0]
        finally:
            cap.release()
        if not ok:
            raise InvalidInputError("video could not be decoded (no readable frames)")

    @staticmethod
    def _check_audio_decodes(path: Path) -> None:
        """Decode the first second with ffmpeg; a damaged stream fails here."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:  # ffprobe passed, so this is unlikely; skip rather than block
            logger.warning("ffmpeg not found; skipping audio decode check")
            return
        cmd = [ffmpeg, "-v", "error", "-t", "1", "-i", str(path), "-f", "null", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        # Only the exit code counts: real-world MP3s often log harmless glitches.
        if result.returncode != 0:
            raise InvalidInputError("audio could not be decoded")

    @staticmethod
    def _cv2_duration(path: Path) -> float | None:
        """Fallback duration = frame count / fps, for containers without a duration header."""
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            cap.release()
        return frames / fps if fps and fps > 0 and frames > 0 else None

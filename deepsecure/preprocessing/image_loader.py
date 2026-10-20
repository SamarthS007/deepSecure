"""Layer 2a (images): open still images safely and identically everywhere.

Both the validator and the frame extractor load images through
:func:`load_image_rgb`, so "what was checked" and "what was analysed" are
exactly the same pixels. It handles the things that make real-world photos
differ from video frames:

* **Format sniffing**: Pillow identifies the format from the file's *contents*
  (magic bytes), not its extension, so an MP4 renamed to ``.jpg`` is rejected.
* **Decompression bombs**: a 50 KB PNG can declare 100 000 x 100 000 pixels
  (~30 GB of RAM once decoded). We read the size from the header and refuse
  anything above ``max_megapixels`` *before* decoding.
* **EXIF orientation**: phones store photos sideways and add an "Orientation"
  tag telling viewers to rotate them. Face detectors expect upright faces, so we
  apply the tag (``ImageOps.exif_transpose``).
* **Colour modes**: greyscale, palette and transparent (RGBA) images are all
  converted to 8-bit RGB; transparent areas are composited onto white.
* **Animated files** (animated WebP): only the first frame is analysed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from deepsecure.exceptions import InvalidInputError

logger = logging.getLogger(__name__)

# Formats as Pillow names them after sniffing the contents.
ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP"})
EXIF_ORIENTATION = 0x0112
SIDEWAYS_ORIENTATIONS = {5, 6, 7, 8}  # values that mean "rotate by 90° to display"


@dataclass(frozen=True)
class ImageInfo:
    """What the header says about an image (read before decoding)."""

    format: str
    width: int
    height: int
    n_frames: int


def _open_checked(path: Path, max_megapixels: float) -> tuple[Image.Image, ImageInfo]:
    """Open lazily (header only), check format and size, return the still-undecoded image."""
    try:
        img = Image.open(path)
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidInputError("image is corrupted or not a supported image format") from exc
    if img.format not in ALLOWED_FORMATS:
        img.close()
        raise InvalidInputError(f"image format {img.format or 'unknown'} is not supported")
    width, height = img.size
    if width * height > max_megapixels * 1_000_000:
        img.close()
        raise InvalidInputError(
            f"image is {width}x{height} ({width * height / 1e6:.0f} MP); "
            f"the limit is {max_megapixels:g} MP"
        )
    if img.getexif().get(EXIF_ORIENTATION) in SIDEWAYS_ORIENTATIONS:
        width, height = height, width  # report the upright size, as it will be analysed
    info = ImageInfo(img.format, width, height, getattr(img, "n_frames", 1))
    return img, info


def inspect_image(path: Path | str, max_megapixels: float) -> ImageInfo:
    """Fully decode the image once (catches truncated files) and return its header info."""
    path = Path(path)
    img, info = _open_checked(path, max_megapixels)
    try:
        img.load()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidInputError("image is corrupted or truncated (could not be decoded)") from exc
    finally:
        img.close()
    return info


def load_image_rgb(path: Path | str, max_megapixels: float) -> np.ndarray:
    """Decode an image as an upright ``(H, W, 3)`` uint8 RGB array.

    Raises:
        InvalidInputError: unsupported format, too many pixels, or undecodable data.
    """
    path = Path(path)
    img, info = _open_checked(path, max_megapixels)
    try:
        if info.n_frames > 1:
            logger.info("%s has %d frames; analysing the first", path.name, info.n_frames)
        img.seek(0)
        upright = ImageOps.exif_transpose(img)  # returns a decoded copy
        if upright.mode in ("RGBA", "LA", "PA") or (
            upright.mode == "P" and "transparency" in upright.info
        ):
            rgba = upright.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            upright = Image.alpha_composite(white, rgba)
        rgb = np.asarray(upright.convert("RGB"), dtype=np.uint8)
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidInputError("image is corrupted or truncated (could not be decoded)") from exc
    finally:
        img.close()
    return np.ascontiguousarray(rgb)

"""Ask ffprobe what's inside a media file (streams and duration) without decoding it.

ffprobe ships with ffmpeg. It reads the file's container headers, so it's fast
and works for every format we accept (mp4, mov, mkv, webm, wav, mp3, m4a...).
If ffprobe can't parse a file, the file is corrupted or isn't media at all.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from deepsecure.exceptions import AssetMissingError, InvalidInputError


@dataclass(frozen=True)
class ProbeResult:
    """What ffprobe found in a file."""

    format_name: str
    duration_s: float | None
    has_video: bool
    has_audio: bool


@lru_cache(maxsize=1)
def ffprobe_path() -> str:
    """Location of the ffprobe binary, or a clear error telling you how to install it."""
    path = shutil.which("ffprobe")
    if path is None:
        raise AssetMissingError("ffprobe not found on PATH. Install ffmpeg: brew install ffmpeg")
    return path


def probe(path: Path, timeout_s: float = 30.0) -> ProbeResult:
    """Inspect ``path`` with ffprobe.

    Raises:
        InvalidInputError: if ffprobe can't read the file (corrupted or not media).
    """
    cmd = [ffprobe_path(), "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]  # fmt: skip
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise InvalidInputError(f"timed out while inspecting {Path(path).name}") from exc
    if result.returncode != 0:
        last = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown"
        detail = last.split(": ", 1)[-1]  # drop ffprobe's "<path>: " prefix
        raise InvalidInputError(f"file is corrupted or not a media file ({detail})")

    info = json.loads(result.stdout or "{}")
    streams = info.get("streams", [])
    # Cover art in an mp3/m4a is a one-picture "video" stream; it isn't real video.
    has_video = any(
        s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        for s in streams
    )
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    durations = [info.get("format", {}).get("duration")]
    durations += [s.get("duration") for s in streams]
    parsed = [float(d) for d in durations if d not in (None, "N/A")]
    return ProbeResult(
        format_name=info.get("format", {}).get("format_name", ""),
        duration_s=max(parsed) if parsed else None,
        has_video=has_video,
        has_audio=has_audio,
    )

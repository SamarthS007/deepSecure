"""Create small synthetic demo clips in data/samples/ (no dataset needed).

Usage (from the repo root):
    uv run python scripts/make_sample_clips.py

Produces:
    shapes_no_audio.mp4    silent video, no face        -> visual/lip-sync/audio unavailable
    shapes_with_tone.mp4   video + tone audio, no face  -> audio only (needs ffmpeg)
    tone_speechlike.wav    pulsing tone, ~speech rhythm -> audio-only input
    white_noise.wav        white noise                  -> audio-only, "unnatural" input
    too_short.wav          0.5 s clip                   -> rejected by the 1 s minimum
    corrupted.mp4          random bytes                 -> rejected by the integrity check
    talking_face_sync.mp4  face + tone, mouth in sync   -> all three modalities available
    talking_face_lag500ms.mp4  same, mouth 0.5 s late   -> a lip-sync error (needs ffmpeg)
    face_photo.jpg         still photo of the face      -> image input: visual only
    face_photo_phone.jpg   same, stored sideways + EXIF -> tests orientation handling
    no_face.png            still, no face               -> image with nothing to analyse
    corrupted.jpg          truncated JPEG               -> rejected by the decode check

The talking-face clips animate a public-domain NASA portrait (see
deepsecure/utils/assets/PROVENANCE.md); they are synthetic test signals, not real
speech. For a realistic demo, add your own short selfie video (with consent).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deepsecure.config.settings import get_settings
from deepsecure.utils.synthetic_media import (
    mux_audio_into_video,
    write_corrupted_file,
    write_face_photo,
    write_no_face_image,
    write_noise_wav,
    write_shapes_video,
    write_talking_face_clip,
    write_tone_wav,
    write_truncated_copy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=get_settings().samples_dir,
        help="destination folder (default: data/samples)",
    )
    parser.add_argument("--duration", type=float, default=4.0, help="clip length in seconds")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    made = [
        write_shapes_video(out / "shapes_no_audio.mp4", duration_s=args.duration),
        write_tone_wav(out / "tone_speechlike.wav", duration_s=args.duration),
        write_noise_wav(out / "white_noise.wav", duration_s=args.duration),
        write_tone_wav(out / "too_short.wav", duration_s=0.5),
        write_corrupted_file(out / "corrupted.mp4"),
        write_face_photo(out / "face_photo.jpg"),
        write_face_photo(out / "face_photo_phone.jpg", exif_rotated=True),
        write_no_face_image(out / "no_face.png"),
    ]
    made.append(write_truncated_copy(out / "face_photo.jpg", out / "corrupted.jpg"))
    try:
        made.append(
            mux_audio_into_video(
                out / "shapes_no_audio.mp4",
                out / "tone_speechlike.wav",
                out / "shapes_with_tone.mp4",
            )
        )
        # Irregular, speech-like rhythm: a periodic one would look in sync at some lags.
        made.append(
            write_talking_face_clip(out / "talking_face_sync.mp4", args.duration, rhythm="speech")
        )
        made.append(
            write_talking_face_clip(
                out / "talking_face_lag500ms.mp4", args.duration, lag_s=0.5, rhythm="speech"
            )
        )
    except (RuntimeError, OSError) as exc:
        print(f"[warn] skipped the clips that need ffmpeg: {exc}")

    for path in made:
        print(f"[ok  ] {path} ({path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Unpack RAVDESS talking-face videos for lip-sync calibration and a held-out real-speech test set.

Usage (from the repo root, after `bash scripts/fetch_datasets.sh ravdess`):
    uv run python scripts/import_ravdess.py

RAVDESS (Livingstone & Russo, 2018; CC BY-NC-SA 4.0) has 24 professional actors saying two
sentences in 8 emotions, filmed frontally in 720p with sound. File names encode
modality-vocal-emotion-intensity-statement-repetition-actor; modality 01 is audio+video
(02 is the same clip without sound, which we skip).

Actors are split by person: by default 01-06 go to data/raw/lipsync_calibration/Actor_XX/
(used by calibrate_lipsync.py) and 07-08 to data/raw/ravdess_test/Actor_XX/ (never used for
calibration; real talking faces for evaluation).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from deepsecure.config.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=s.data_dir / "raw" / "ravdess")
    parser.add_argument("--test-actors", nargs="+", default=["07", "08"])
    args = parser.parse_args(argv)

    zips = sorted(args.src.glob("Video_Speech_Actor_*.zip"))
    if not zips:
        print(f"[error] no Video_Speech_Actor_*.zip in {args.src}")
        print("        run: bash scripts/fetch_datasets.sh ravdess")
        return 1
    counts: dict[str, int] = {}
    for z in zips:
        actor = z.stem.rsplit("_", 1)[1]
        dest = (
            s.data_dir
            / "raw"
            / ("ravdess_test" if actor in args.test_actors else "lipsync_calibration")
            / f"Actor_{actor}"
        )
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                base = name.rsplit("/", 1)[-1]
                if not base.startswith("01-") or not base.endswith(".mp4"):
                    continue  # keep audio+video clips only
                target = dest / base
                if not target.is_file():
                    target.write_bytes(zf.read(name))
                counts[dest.parent.name] = counts.get(dest.parent.name, 0) + 1
        print(f"[ok  ] actor {actor} -> {dest}")
    for folder, n in counts.items():
        print(f"[info] {folder}: {n} audio+video clips")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download the MediaPipe models and the pretrained AASIST voice model (and optionally FF++ splits).

Usage (from the repo root):
    uv run python scripts/download_assets.py            # skip files that already exist
    uv run python scripts/download_assets.py --force    # re-download everything
    uv run python scripts/download_assets.py --ff-splits # + official FaceForensics++ splits

After downloading, the script loads each model once to prove the file works. AASIST's
weights (1.3 MB, MIT licence) go to models/audio/aasist/ and are checked against a pinned
SHA-256 hash.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import torch

from deepsecure.config.settings import get_settings
from deepsecure.data.faceforensics import OFFICIAL_SPLITS_URL
from deepsecure.models import aasist_model
from deepsecure.preprocessing.mediapipe_assets import (
    ASSETS,
    asset_path,
    create_face_detector,
    create_face_landmarker,
)


def download(url: str, dest: Path) -> int:
    """Download ``url`` to ``dest`` via a temporary ``.part`` file; return the size in bytes.

    Writing to ``.part`` first means an interrupted download never leaves a
    half-written file that looks valid.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as fh:
        while chunk := response.read(1 << 16):
            fh.write(chunk)
    tmp.replace(dest)
    return dest.stat().st_size


def verify() -> None:
    """Load both models once; raises if a file is unusable."""
    with create_face_detector():
        pass
    with create_face_landmarker():
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=get_settings().mediapipe_dir,
        help="destination folder (default: models/mediapipe)",
    )
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument(
        "--ff-splits",
        action="store_true",
        help="also fetch the official FaceForensics++ train/val/test.json split lists "
        "into data/raw/faceforensics/splits/ (MIT-licensed, from the FF++ GitHub repo)",
    )
    args = parser.parse_args()

    if args.ff_splits:
        splits_dir = get_settings().data_dir / "raw" / "faceforensics" / "splits"
        for split in ("train", "val", "test"):
            dest = splits_dir / f"{split}.json"
            if dest.is_file() and not args.force:
                print(f"[skip] {dest.name} already present")
                continue
            download(OFFICIAL_SPLITS_URL.format(split=split), dest)
            print(f"[ok  ] {dest}")

    for asset in ASSETS.values():
        dest = asset_path(asset.key, args.out)
        if dest.is_file() and dest.stat().st_size > 0 and not args.force:
            print(f"[skip] {dest.name} already present ({dest.stat().st_size / 1e6:.2f} MB)")
            continue
        print(f"[get ] {asset.filename}  <- {asset.url}")
        size = download(asset.url, dest)
        print(f"[ok  ] {dest} ({size / 1e6:.2f} MB)")

    if args.out == get_settings().mediapipe_dir:
        verify()
        print("[ok  ] Both MediaPipe models load correctly with the Tasks API.")

    weights = get_settings().models_dir / "audio" / "aasist" / aasist_model.WEIGHTS_FILE
    if weights.is_file() and not args.force:
        print(f"[skip] {weights.name} already present")
    else:
        print(f"[get ] {weights.name}  <- {aasist_model.WEIGHTS_URL}")
        download(aasist_model.WEIGHTS_URL, weights)
    if aasist_model.sha256_file(weights) != aasist_model.WEIGHTS_SHA256:
        weights.unlink()
        print("[FAIL] AASIST weights failed the SHA-256 check and were deleted; try again")
        return 1
    aasist_model.AASIST(weights, torch.device("cpu"))
    print(f"[ok  ] {weights} (SHA-256 verified, loads with weights_only=True)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

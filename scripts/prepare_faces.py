"""Turn a split CSV of videos/images into face-crop PNGs for training the visual detector.

Usage (from the repo root):
    uv run python scripts/prepare_faces.py                                  # FF++ splits
    uv run python scripts/prepare_faces.py --max-per-class 100              # quick subset
    uv run python scripts/prepare_faces.py --synthetic 40                   # toy data, no dataset

Reads data/processed/splits/faceforensics.csv (from make_splits.py), runs the same
FrameExtractor as the app (largest face, square +20 %, 299 x 299) on up to
--crops-per-video frames per video (1 per second by default), and writes
    data/processed/faces/{split}/{real,fake}/<video-id>_<n>.png
    data/processed/faces/index.csv   (path, label, split, group, video, dataset)
PNG is lossless, so preparing the data adds no compression artifacts of its own.
Re-running skips videos that already have crops, so an interrupted run can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from deepsecure.config.settings import get_settings
from deepsecure.data.splits import find_leakage
from deepsecure.exceptions import DeepSecureError
from deepsecure.preprocessing.frame_extractor import FrameExtractor
from deepsecure.utils.synthetic_media import synthetic_face_crops

LABEL_DIR = {0: "real", 1: "fake"}


def video_key(rel_path: str) -> str:
    """Unique, file-system-safe id: FF++ reuses names like 000_003.mp4 across methods."""
    digest = hashlib.sha1(rel_path.encode()).hexdigest()[:8]
    return f"{Path(rel_path).stem}_{digest}"


def save_crops(crops: np.ndarray, folder: Path, key: str) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, crop in enumerate(crops):
        path = folder / f"{key}_{i:02d}.png"
        Image.fromarray(crop).save(path)
        paths.append(path)
    return paths


def prepare_real(df: pd.DataFrame, out: Path, crops_per_video: int) -> list[dict]:
    s = get_settings()
    settings = s.model_copy(
        update={"max_frames": crops_per_video, "frame_sample_fps": s.visual_prep_fps}
    )
    rows: list[dict] = []
    with FrameExtractor(settings) as fx:
        for n, rec in enumerate(df.itertuples(index=False), start=1):
            key = video_key(str(rec.path))
            folder = out / rec.split / LABEL_DIR[int(rec.label)]
            existing = sorted(folder.glob(f"{key}_*.png"))
            if existing:
                paths = existing
            else:
                try:
                    batch = fx.extract(s.data_dir / rec.path, with_lipsync=False)
                except DeepSecureError as exc:
                    print(f"[skip] {rec.path}: {getattr(exc, 'reason', exc)}")
                    continue
                if batch.faces_found == 0:
                    print(f"[skip] {rec.path}: no face found")
                    continue
                paths = save_crops(batch.face_crops, folder, key)
            for p in paths:
                rows.append({"path": str(p.relative_to(out)), "label": int(rec.label),
                             "split": rec.split, "group": rec.group, "video": key,
                             "dataset": getattr(rec, "dataset", "")})  # fmt: skip
            if n % 25 == 0:
                print(f"[info] {n}/{len(df)} videos, {len(rows)} crops")
    return rows


def prepare_synthetic(n_videos: int, out: Path, crops_per_video: int = 4) -> list[dict]:
    """``n_videos`` toy "videos" per class; every 4th one goes to val."""
    rows: list[dict] = []
    for label in (0, 1):
        for v in range(n_videos):
            split = "val" if v % 4 == 3 else "train"
            key = f"syn{label}_{v:03d}"
            crops = synthetic_face_crops(crops_per_video, fake=bool(label), seed=1000 * label + v)
            for p in save_crops(crops, out / split / LABEL_DIR[label], key):
                rows.append({"path": str(p.relative_to(out)), "label": label, "split": split,
                             "group": key, "video": key, "dataset": "synthetic"})  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_csv = s.data_dir / "processed" / "splits" / "faceforensics.csv"
    parser.add_argument("--splits-csv", type=Path, default=default_csv)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--crops-per-video", type=int, default=s.visual_crops_per_video)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="per split: keep at most N real and N fake videos")  # fmt: skip
    parser.add_argument("--synthetic", type=int, default=0, help="N toy videos per class")
    parser.add_argument("--seed", type=int, default=s.split_seed)
    args = parser.parse_args(argv)

    if args.synthetic:
        out = args.out or s.data_dir / "processed" / "faces_synthetic"
        rows = prepare_synthetic(args.synthetic, out)
    else:
        out = args.out or s.data_dir / "processed" / "faces"
        if not args.splits_csv.is_file():
            print(f"[error] {args.splits_csv} not found: run scripts/make_splits.py first")
            return 1
        df = pd.read_csv(args.splits_csv)
        df = df[df["split"].isin(args.splits)]
        if "exists" in df.columns:
            df = df[df["exists"].astype(bool)]
        if args.max_per_class:
            shuffled = df.sample(frac=1.0, random_state=args.seed)
            df = shuffled.groupby(["split", "label"]).head(args.max_per_class)
        print(f"[info] {len(df)} videos -> {out}")
        rows = prepare_real(df, out, args.crops_per_video)

    index = pd.DataFrame(rows)
    if index.empty:
        print("[error] no crops were produced")
        return 1
    problems = find_leakage(index)
    if problems:
        print("[error] leakage between splits:\n  " + "\n  ".join(problems[:10]))
        return 1
    out.mkdir(parents=True, exist_ok=True)
    index.to_csv(out / "index.csv", index=False)
    summary = index.groupby(["split", "label"]).agg(crops=("path", "size"),
                                                    videos=("video", "nunique"))  # fmt: skip
    print(summary.to_string())
    print(f"[ok  ] {len(index)} crops; index at {out / 'index.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

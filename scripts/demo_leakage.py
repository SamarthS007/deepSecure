"""Show, with numbers, how data leakage inflates accuracy (no dataset needed).

Usage (from the repo root):
    uv run python scripts/demo_leakage.py
    uv run python scripts/demo_leakage.py --signal 0.0     # labels carry no real signal at all

How the toy data is built: there are N "videos", half real and half fake. Each
video has its own random "identity" vector (think: the person's face, lighting,
camera). Every frame of a video = identity + small per-frame noise + a *weak*
real-vs-fake signal of strength --signal. Then we train the same 1-nearest-neighbour
classifier twice:

  * frame-level split (WRONG): frames are shuffled into train/test individually,
    so each test frame has near-identical sibling frames from the same video in train;
  * video-level split (RIGHT): whole videos go to train or test (our group_split).

The gap between the two accuracies is pure leakage: the model memorised the
*videos*, not the difference between real and fake.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from deepsecure.config.settings import get_settings
from deepsecure.data.splits import group_split


def make_toy_frames(
    n_videos: int, frames_per_video: int, dim: int, signal: float, seed: int
) -> pd.DataFrame:
    """Build frame-level features with a strong per-video identity and a weak label signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_videos):
        label = v % 2
        identity = rng.normal(0.0, 1.0, dim)
        for _ in range(frames_per_video):
            x = identity + rng.normal(0.0, 0.1, dim)
            x[0] += signal * (1 if label else -1)
            rows.append({"group": f"video_{v:03d}", "label": label, "x": x})
    df = pd.DataFrame(rows)
    df["path"] = [f"{g}/frame_{i}" for i, g in enumerate(df["group"])]
    return df


def accuracy(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Train a 1-NN classifier on ``train`` frames and return accuracy on ``test`` frames."""
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(np.stack(train["x"]), train["label"])
    return float((clf.predict(np.stack(test["x"])) == test["label"]).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", type=int, default=100)
    parser.add_argument("--frames", type=int, default=30, help="frames per video")
    parser.add_argument("--dim", type=int, default=16, help="feature dimensions")
    parser.add_argument("--signal", type=float, default=0.25, help="real-vs-fake signal strength")
    parser.add_argument("--seeds", type=int, default=20, help="repeat with this many seeds")
    parser.add_argument("--out", type=Path, default=get_settings().reports_dir / "leakage_demo.csv")
    args = parser.parse_args()

    results = []
    for seed in range(args.seeds):
        df = make_toy_frames(args.videos, args.frames, args.dim, args.signal, seed)

        # WRONG: frame-level split, 85% train / 15% test, frames shuffled individually.
        tr, te = train_test_split(df, test_size=0.15, random_state=seed, stratify=df["label"])
        leaky = accuracy(tr, te)

        # RIGHT: video-level split; val is merged into train here to keep the comparison fair.
        split = group_split(df, ratios=(0.70, 0.15, 0.15), seed=seed)
        honest = accuracy(split[split["split"] != "test"], split[split["split"] == "test"])
        results.append(
            {"seed": seed, "frame_level_split_acc": leaky, "video_level_split_acc": honest}
        )

    table = pd.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"{args.videos} videos x {args.frames} frames, signal={args.signal}, 1-NN classifier")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nmean accuracy  frame-level split (leaky):  {table.iloc[:, 1].mean():.3f}")
    print(f"mean accuracy  video-level split (honest): {table.iloc[:, 2].mean():.3f}")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Calibrate (train) the lip-sync baseline from REAL talking-face clips only.

Usage (from the repo root):
    uv run python scripts/calibrate_lipsync.py                      # data/raw/lipsync_calibration/
    uv run python scripts/calibrate_lipsync.py --videos path/to/clips
    uv run python scripts/calibrate_lipsync.py --synthetic 12       # demo on generated clips

No labels needed (self-supervised): every real clip is a positive (in sync), and
negatives are made from it by shifting its audio 0.3-1.5 s or giving it another
clip's audio. Clips are split into train/val *by clip*, so a clip's negatives never
leak across the split. The val metrics are printed and stored in the file.

Use clips with one person talking to the camera, with sound (a phone selfie video is
ideal; get consent). Do NOT use the internal test set (data/raw/internal): that's
reserved for the final evaluation. FaceForensics++ videos are normally used without
audio, so they can't be used here.

Real calibrations go to models/lipsync/lipsync_baseline_calibration.joblib (loaded
by the app). --synthetic runs write a separate file marked data_source="synthetic",
which never switches off demo mode.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from deepsecure.config.settings import get_settings
from deepsecure.exceptions import DeepSecureError
from deepsecure.models.lipsync_model import (
    CALIBRATION_FILE,
    FEATURE_NAMES,
    LipSyncDetector,
    calibration_rows,
    fit_calibration,
    save_calibration,
    uncalibrated_fake_prob,
)
from deepsecure.preprocessing.audio_extractor import AudioExtractor
from deepsecure.preprocessing.frame_extractor import FrameExtractor
from deepsecure.utils.synthetic_media import write_talking_face_clip


def find_videos(folder: Path, extensions: tuple[str, ...]) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in extensions)


def make_synthetic(n: int, folder: Path, duration_s: float = 4.0) -> list[Path]:
    """In-sync synthetic clips, each with its own irregular rhythm (seed = clip number)."""
    folder.mkdir(parents=True, exist_ok=True)
    return [
        write_talking_face_clip(
            folder / f"synthetic_{i:02d}.mp4", duration_s, rhythm="speech", seed=i
        )
        for i in range(n)
    ]


RAVDESS_NAME = re.compile(r"^\d{2}-\d{2}-\d{2}-\d{2}-(\d{2})-\d{2}-\d{2}\.mp4$")


def sentence_keys(names: list[str]) -> list[str] | None:
    """For RAVDESS files (modality-vocal-emotion-intensity-STATEMENT-repetition-actor.mp4) the
    statement code, so swapped audio always says the *other* sentence; otherwise None."""
    matches = [RAVDESS_NAME.match(n) for n in names]
    return [m.group(1) for m in matches] if all(matches) else None


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", type=Path, default=s.data_dir / "raw" / "lipsync_calibration")
    parser.add_argument("--synthetic", type=int, default=0, help="generate N synthetic clips")
    parser.add_argument("--out", type=Path, default=None, help="output .joblib")
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=s.split_seed)
    parser.add_argument(
        "--group-by",
        choices=["parent", "clip"],
        default="parent",
        help="keep each speaker folder (parent) or each clip on one side of the split",
    )
    args = parser.parse_args(argv)

    if args.synthetic:
        source = "synthetic"
        videos = make_synthetic(args.synthetic, s.data_dir / "processed" / "lipsync_synthetic")
        out = args.out or s.models_dir / "lipsync" / "lipsync_calibration_synthetic.joblib"
    else:
        source = str(args.videos)
        if not args.videos.is_dir():
            print(f"[error] folder not found: {args.videos} (see DOCUMENTATION §8.6)")
            return 1
        videos = find_videos(args.videos, s.video_extensions)
        out = args.out or s.models_dir / "lipsync" / CALIBRATION_FILE
    if len(videos) < 4:
        print(f"[error] need at least 4 clips, found {len(videos)} in {args.videos}")
        return 1

    clips, names, parents = [], [], []
    audio_x = AudioExtractor(s)
    with FrameExtractor(s) as fx:
        for path in videos:
            try:
                clips.append((fx.extract(path), audio_x.extract(path)))
                names.append(path.name)
                parents.append(path.parent.name)
            except DeepSecureError as exc:
                print(f"[skip] {path.name}: {getattr(exc, 'reason', exc)}")
    print(f"[info] {len(clips)} usable clips of {len(videos)}")

    detector = LipSyncDetector(s)
    rng = np.random.default_rng(args.seed)
    groups_in = parents if args.group_by == "parent" else list(range(len(clips)))
    rows, labels, groups = calibration_rows(
        clips, detector, rng, groups=groups_in, swap_keys=sentence_keys(names)
    )
    y, g = np.array(labels), np.array(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y, g))
    print(
        f"[info] rows: train {len(train_idx)} / val {len(val_idx)} "
        f"({args.group_by} groups {len(set(g[train_idx]))} / {len(set(g[val_idx]))})"
    )
    if len(set(y[val_idx])) < 2 or len(set(y[train_idx])) < 2:
        print("[error] a split has only one class; add more clips")
        return 1

    model = fit_calibration([rows[i] for i in train_idx], list(y[train_idx]))
    x_val = np.array([[rows[i][k] for k in FEATURE_NAMES] for i in val_idx])
    p_cal = model.predict_proba(x_val)[:, 1]
    p_raw = np.array([uncalibrated_fake_prob(rows[i], s) for i in val_idx])
    metrics = {
        "val_accuracy": float(accuracy_score(y[val_idx], p_cal >= 0.5)),
        "val_roc_auc": float(roc_auc_score(y[val_idx], p_cal)),
        "val_accuracy_uncalibrated": float(accuracy_score(y[val_idx], p_raw >= 0.5)),
        "val_roc_auc_uncalibrated": float(roc_auc_score(y[val_idx], p_raw)),
    }
    for k, v in metrics.items():
        print(f"[val ] {k:28} {v:.3f}")

    save_calibration(
        model,
        out,
        data_source=source,
        n_clips=len(clips),
        n_rows=len(rows),
        created_at=datetime.now(UTC).isoformat(),
        seed=args.seed,
        **metrics,
    )
    print(f"[ok  ] saved {out} (data_source={source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Train the audio (spoofed-speech) detector on ASVspoof 2019 LA summary features.

Usage (from the repo root):
    uv run python scripts/train_audio.py                         # full train + dev splits
    uv run python scripts/train_audio.py --max-per-class 2000    # quicker subset
    uv run python scripts/train_audio.py --synthetic 200         # toy voices, no dataset

1. Reads data/processed/splits/asvspoof2019_la.csv (make_splits.py): official
   train -> "train", dev -> "val". The eval partition ("test") is NOT touched here.
2. Extracts the 170-number summary vector of every file with the app's own
   AudioExtractor (in parallel), cached in data/processed/audio_features/ so a
   re-run only extracts new files.
3. Fits logistic regression and an MLP on train (balanced weights), keeps the one with
   the lower dev EER, and saves models/audio/audio_classifier.joblib with its
   metrics and the dev EER threshold. The comparison goes to reports/.

--synthetic writes audio_classifier_synthetic.joblib (data_source="synthetic"),
which never ends demo mode.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from deepsecure.config.settings import Settings, get_settings
from deepsecure.models.audio_model import (
    CLASSIFIER_FILE,
    save_classifier,
    select_model,
    summary_features,
)
from deepsecure.preprocessing.audio_extractor import AudioExtractor
from deepsecure.utils.synthetic_media import synthetic_voice


def synthetic_split(n_per_class: int, seed_offset: int, s: Settings):
    ax = AudioExtractor(s)
    rows, labels = [], []
    for label in (0, 1):
        for i in range(n_per_class):
            wave = synthetic_voice(spoof=bool(label), seed=seed_offset + 10_000 * label + i)
            rows.append(ax.features_from_waveform(wave, s.audio_sr).summary_vector)
            labels.append(label)
    return np.stack(rows).astype(np.float64), np.array(labels)


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_csv = s.data_dir / "processed" / "splits" / "asvspoof2019_la.csv"
    parser.add_argument("--splits-csv", type=Path, default=default_csv)
    parser.add_argument("--max-per-class", type=int, default=None, help="per split")
    parser.add_argument("--synthetic", type=int, default=0, help="N toy voices per class")
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=s.split_seed)
    args = parser.parse_args(argv)

    if args.synthetic:
        source = "synthetic"
        x_tr, y_tr = synthetic_split(args.synthetic, 0, s)
        x_va, y_va = synthetic_split(max(args.synthetic // 4, 10), 5_000, s)
        out = args.out or s.models_dir / "audio" / "audio_classifier_synthetic.joblib"
    else:
        if not args.splits_csv.is_file():
            print(f"[error] {args.splits_csv} not found: run scripts/make_splits.py first")
            return 1
        df = pd.read_csv(args.splits_csv)
        if "exists" in df.columns:
            df = df[df["exists"].astype(bool)]
        df = df[df["split"].isin(["train", "val"])]
        if args.max_per_class:
            shuffled = df.sample(frac=1.0, random_state=args.seed)
            df = shuffled.groupby(["split", "label"]).head(args.max_per_class)
        cache_dir = s.data_dir / "processed" / "audio_features"
        x_tr, y_tr = summary_features(df, "train", s, args.n_jobs, cache_dir)
        x_va, y_va = summary_features(df, "val", s, args.n_jobs, cache_dir)
        source = str(args.splits_csv)
        out = args.out or s.models_dir / "audio" / CLASSIFIER_FILE

    print(f"[info] train {len(y_tr)} ({int(y_tr.sum())} spoof), val {len(y_va)} "
          f"({int(y_va.sum())} spoof)")  # fmt: skip
    best, model, results = select_model(x_tr, y_tr, x_va, y_va, s, args.seed)
    table = pd.DataFrame(results).T.rename_axis("model")
    print(table.round(4).to_string())

    chosen = results[best]
    save_classifier(
        model, out, s, data_source=source, model_name=best, n_train=len(y_tr), n_val=len(y_va),
        val_eer=chosen["eer"], val_eer_threshold=chosen["eer_threshold"],
        val_roc_auc=chosen["roc_auc"], val_accuracy=chosen["accuracy"], seed=args.seed,
    )  # fmt: skip
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    report = s.reports_dir / f"audio_model_selection{'_synthetic' if args.synthetic else ''}.csv"
    table.to_csv(report)
    print(f"[ok  ] kept {best} (val EER {chosen['eer']:.4f}); saved {out} "
          f"(data_source={source}); comparison {report}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())

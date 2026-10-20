"""Compare learned fusion with the weighted baseline (grouped cross-validation), optionally save it.

Usage (from the repo root):
    uv run python scripts/train_fusion.py                       # reports/scores_internal_all.csv
    uv run python scripts/train_fusion.py --save                # also fit on all rows and save
    uv run python scripts/train_fusion.py --synthetic 400       # toy score table

Input: the per-file scores written by evaluate_models.py for the internal set (one
row per file: label, media_type, group, visual/audio/lipsync probabilities and
confidences, NaN = unavailable). These are the only rows in the project with all
three modalities *and* a label, so the internal set doubles as fusion training data.
To keep the comparison honest, the learned head is scored only on folds it wasn't
fitted on (StratifiedGroupKFold: a person's files never sit on both sides).

The weighted baseline needs no fitting, so its numbers on the same rows are directly
comparable. Use learned fusion (DEEPSECURE_FUSION_METHOD=learned) only if it beats
the baseline here by a clear margin; with tens of files it usually can't be trusted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from deepsecure.config.settings import Settings, get_settings
from deepsecure.fusion.fusion_engine import FusionEngine
from deepsecure.fusion.learned_head import HEAD_FILE, MODALITIES, LearnedFusionHead
from deepsecure.models.metrics import binary_metrics
from deepsecure.schemas import ModalityScore
from deepsecure.utils.synthetic_media import synthetic_fusion_scores


def row_scores(row: pd.Series) -> list[ModalityScore]:
    """Rebuild the ModalityScores of one file from a scores table row."""
    out = []
    for m in MODALITIES:
        p = row.get(m, np.nan)
        if pd.isna(p):
            out.append(ModalityScore(name=m, fake_prob=0.5, confidence=0.0, available=False))
        else:
            conf = row.get(f"{m}_conf", np.nan)
            out.append(
                ModalityScore(
                    name=m,
                    fake_prob=float(p),
                    trained=True,
                    confidence=float(conf) if not pd.isna(conf) else 1.0,
                )
            )
    return out


def weighted_probs(df: pd.DataFrame, s: Settings) -> np.ndarray:
    engine = FusionEngine(s.model_copy(update={"fusion_method": "weighted"}))
    return np.array(
        [
            engine.fuse(row_scores(r), r.get("media_type", "video")).fused_fake_prob
            for _, r in df.iterrows()
        ]
    )


def cross_validated(df: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    """Out-of-fold learned-fusion probabilities for every row."""
    y = df["label"].to_numpy()
    groups = df["group"].to_numpy() if "group" in df else np.arange(len(df))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = np.full(len(df), np.nan)
    for train_idx, test_idx in splitter.split(df, y, groups):
        head = LearnedFusionHead().fit(df.iloc[train_idx])
        out[test_idx] = head.predict_frame(df.iloc[test_idx])
    return out


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, default=s.reports_dir / "scores_internal_all.csv")
    parser.add_argument("--synthetic", type=int, default=0, help="N toy files")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--save", action="store_true", help="fit on all rows and save the head")
    parser.add_argument("--seed", type=int, default=s.split_seed)
    args = parser.parse_args(argv)

    if args.synthetic:
        df, source = synthetic_fusion_scores(args.synthetic, args.seed), "synthetic"
        out_dir, name = s.reports_dir / "synthetic", "fusion_head_synthetic.joblib"
    else:
        if not args.scores.is_file():
            print(
                f"[skip] {args.scores} not found: run evaluate_models.py on the internal set "
                "(with trained detectors). Writing a TBD row."
            )
            s.reports_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "method": "weighted / learned (CV)",
                        "status": "skipped: no internal-set scores yet — TBD",
                    }
                ]
            ).to_csv(s.reports_dir / "fusion_comparison.csv", index=False)
            return 0
        df, source = pd.read_csv(args.scores), str(args.scores)
        out_dir, name = s.reports_dir, HEAD_FILE
    if "status" in df:
        df = df[df["status"] == "ok"].reset_index(drop=True)

    per_class = df.groupby("label")["group" if "group" in df else "path"].nunique()
    folds = int(min(args.folds, per_class.min() if len(per_class) == 2 else 0))
    if folds < 2:
        print(
            f"[error] need >= 2 identity groups per class for cross-validation; have "
            f"{per_class.to_dict()}"
        )
        return 1

    y = df["label"].to_numpy()
    candidates = {
        "weighted": weighted_probs(df, s),
        "learned (CV)": cross_validated(df, folds, args.seed),
    }
    for m in MODALITIES:  # single detectors, on the files where each one was available
        candidates[f"{m} only"] = df[m].to_numpy() if m in df else np.full(len(df), np.nan)
    rows = []
    for method, p in candidates.items():
        have = ~np.isnan(p)
        if len(set(y[have].tolist())) < 2:
            rows.append({"method": method, "n": int(have.sum())})
            continue
        m = binary_metrics(y[have], p[have], 0.5)
        rows.append(
            {
                "method": method,
                "n": m["n"],
                "roc_auc": m["roc_auc"],
                "eer": m["eer"],
                "accuracy@0.5": m["accuracy"],
                "f1@0.5": m["f1"],
            }
        )
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "fusion_comparison.csv", index=False)
    print(f"[info] {len(df)} files ({int(y.sum())} fake), {folds}-fold grouped CV")
    print(table.round(4).to_string(index=False))

    if args.save:
        head = LearnedFusionHead().fit(df)
        cv = rows[1]  # "learned (CV)"
        path = head.save(
            s.models_dir / "fusion" / name,
            data_source=source,
            n=len(df),
            cv_roc_auc=cv["roc_auc"],
            cv_eer=cv["eer"],
            folds=folds,
        )
        weights = {k: round(v, 3) for k, v in head.modality_weights().items()}
        print(f"[ok  ] saved {path} (data_source={source}); relative weights {weights}")
    print(f"[ok  ] comparison {out_dir / 'fusion_comparison.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

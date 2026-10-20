"""Fine-tune the visual detector (Xception) on prepared face crops.

Usage (from the repo root):
    uv run python scripts/train_visual.py                          # data/processed/faces/index.csv
    uv run python scripts/train_visual.py --epochs 3 --batch-size 8
    uv run python scripts/prepare_faces.py --synthetic 40 && \\
        uv run python scripts/train_visual.py --synthetic           # toy run, no dataset

Trains on split == "train", selects the best epoch on split == "val" (video-level
ROC-AUC, top-k aggregation as in the app) and never touches "test" (that's Phase 4D).
Writes models/visual/visual_best.pt (loaded by the app) and a per-epoch history CSV in
reports/. A --synthetic run writes visual_synthetic.pt instead, marked
data_source="synthetic", which never ends demo mode.

Measured on this M1 (MPS): a fine-tuning epoch over 240 crops (+80 val) took 39-48 s,
i.e. ~0.15-0.2 s per crop, so 10 000 crops ≈ 25-35 min/epoch.
For larger runs use a CUDA GPU (e.g. Google Colab):
DOCUMENTATION §8.7.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from deepsecure.config.device import describe_device, get_device
from deepsecure.config.settings import get_settings
from deepsecure.models.visual_model import CHECKPOINT_FILE, build_model
from deepsecure.models.visual_train import train_visual


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, default=None, help="index.csv from prepare_faces")
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic toy crops")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--arch", default=s.visual_arch)
    parser.add_argument("--no-pretrained", action="store_true", help="random init (no download)")
    parser.add_argument("--epochs", type=int, default=s.visual_epochs)
    parser.add_argument("--batch-size", type=int, default=s.visual_batch_size)
    parser.add_argument("--lr", type=float, default=s.visual_lr)
    parser.add_argument("--freeze-epochs", type=int, default=s.visual_freeze_epochs)
    parser.add_argument("--patience", type=int, default=s.visual_patience)
    parser.add_argument("--device", default=s.device, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)

    folder = "faces_synthetic" if args.synthetic else "faces"
    index_path = args.index or s.data_dir / "processed" / folder / "index.csv"
    if not index_path.is_file():
        hint = "--synthetic 40" if args.synthetic else "(see DOCUMENTATION §8.7)"
        print(f"[error] {index_path} not found: run scripts/prepare_faces.py {hint}")
        return 1
    index = pd.read_csv(index_path)
    train_df = index[index["split"] == "train"].reset_index(drop=True)
    val_df = index[index["split"] == "val"].reset_index(drop=True)
    shared = set(train_df["group"]) & set(val_df["group"])
    if shared:
        print(f"[error] {len(shared)} identity groups are in both train and val: leakage")
        return 1
    synthetic = bool((index["dataset"] == "synthetic").all())
    source = "synthetic" if synthetic else str(index_path)
    name = "visual_synthetic.pt" if synthetic else CHECKPOINT_FILE
    out = args.out or s.models_dir / "visual" / name

    device = get_device(args.device)
    model, arch = build_model(args.arch, pretrained=not args.no_pretrained,
                              fallback=s.visual_fallback_arch)  # fmt: skip
    n_train, n_val = train_df["video"].nunique(), val_df["video"].nunique()
    print(f"[info] {arch} on {describe_device(device)}; train {len(train_df)} crops / "
          f"{n_train} videos, val {len(val_df)} / {n_val}")  # fmt: skip

    result = train_visual(
        model, arch, train_df, val_df, device, out,
        epochs=args.epochs, lr=args.lr, weight_decay=s.visual_weight_decay,
        freeze_epochs=args.freeze_epochs, patience=args.patience,
        batch_size=args.batch_size, top_k=s.visual_top_k,
        root=index_path.parent, num_workers=args.num_workers,
        metadata={"data_source": source, "n_train_crops": len(train_df),
                  "n_val_crops": len(val_df), "device": str(device),
                  "pretrained": not args.no_pretrained},
    )  # fmt: skip

    history = pd.DataFrame(result.history)
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    hist_path = s.reports_dir / f"visual_train_history{'_synthetic' if synthetic else ''}.csv"
    history.to_csv(hist_path, index=False)
    print(history.round(4).to_string(index=False))
    print(f"[ok  ] best epoch {result.best_epoch} (val video AUC {result.best_val_auc:.4f}); "
          f"saved {result.checkpoint} (data_source={source}); history {hist_path}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fine-tune the published AASIST on In-the-Wild + ASVspoof (DOCUMENTATION §8.17).

Usage (from the repo root, after import_itw.py and make_splits.py --dataset asvspoof):
    uv run python scripts/train_aasist.py                       # defaults: 4 epochs x 600 steps
    uv run python scripts/train_aasist.py --epochs 1 --steps 20 --val-per-class 50   # smoke test

Training data: In-the-Wild *train speakers* and ASVspoof *train*, balanced 4 ways (see
deepsecure/models/aasist_train.py). Model selection and the decision threshold use only
validation data: In-the-Wild *val speakers* and ASVspoof *dev*. Test data is never touched.

Writes models/audio/aasist/AASIST_finetuned.pth (+ finetuned.json: SHA-256, threshold,
validation EERs, recipe) whenever the mean validation EER improves, and
reports/aasist_finetune_history.csv. The app then loads it automatically
(setting aasist_variant = "auto"; "pretrained" switches back).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch

from deepsecure.config.device import describe_device, get_device
from deepsecure.config.settings import get_settings
from deepsecure.models import aasist_model as am
from deepsecure.models.aasist_train import EpochResult, score_clips, train
from deepsecure.models.metrics import compute_eer


def sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None:
        return df
    parts = [g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts)


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600, help="steps per epoch")
    parser.add_argument("--batch", type=int, default=2, help="clips per forward pass (RAM)")
    parser.add_argument("--accum", type=int, default=6, help="forward passes per step")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--val-per-class", type=int, default=600, help="per validation set")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=s.split_seed)
    args = parser.parse_args(argv)

    splits = s.data_dir / "processed" / "splits"
    itw_csv, asv_csv = splits / "in_the_wild.csv", splits / "asvspoof2019_la.csv"
    for csv, how in ((itw_csv, "import_itw.py"), (asv_csv, "make_splits.py --dataset asvspoof")):
        if not csv.is_file():
            print(f"[error] {csv} not found: run scripts/{how}")
            return 1
    itw, asv = pd.read_csv(itw_csv), pd.read_csv(asv_csv)
    if "exists" in asv:
        asv = asv[asv["exists"].astype(bool)]

    def paths(df: pd.DataFrame, split: str, label: int) -> list[tuple[str, int]]:
        part = df[(df["split"] == split) & (df["label"] == label)]
        return [(str(s.data_dir / p), label) for p in part["path"]]

    pools = [paths(itw, "train", 0), paths(itw, "train", 1), paths(asv, "train", 0),
             paths(asv, "train", 1)]  # fmt: skip
    val_sets = {}
    for name, df in (("in_the_wild_val", itw), ("asvspoof_dev", asv)):
        v = sample(df[df["split"] == "val"], args.val_per_class, args.seed)
        val_sets[name] = ([str(s.data_dir / p) for p in v["path"]], v["label"].to_numpy())
    print("[info] training pools (ITW real/fake, ASVspoof real/fake):", [len(p) for p in pools])
    print("[info] validation:", {k: len(v[0]) for k, v in val_sets.items()})

    device = get_device(s.device)
    folder = s.models_dir / "audio" / "aasist"
    model = am.AASIST(folder / am.WEIGHTS_FILE, device).model  # published weights, SHA-checked
    n_train_speakers = int(itw[itw["split"] == "train"]["group"].nunique())

    # epoch 0 = the published model on the same validation sets (the "before" numbers)
    base = {k: float(compute_eer(lab, score_clips(model, p, device))[0])
            for k, (p, lab) in val_sets.items()}  # fmt: skip
    print("[val ] epoch 0 (published AASIST): "
          + ", ".join(f"{k} EER {v:.4f}" for k, v in base.items()), flush=True)  # fmt: skip
    rows = [{"epoch": 0, "train_loss": np.nan, **{f"eer_{k}": v for k, v in base.items()},
             "mean_eer": float(np.mean(list(base.values()))), "seconds": 0.0}]  # fmt: skip
    best = {"mean": np.inf}

    def on_epoch(r: EpochResult, m: torch.nn.Module, val_scores: dict) -> None:
        mean = float(np.mean(list(r.eer.values())))
        rows.append({"epoch": r.epoch, "train_loss": r.train_loss,
                     **{f"eer_{k}": v for k, v in r.eer.items()}, "mean_eer": mean,
                     "seconds": r.seconds})  # fmt: skip
        pd.DataFrame(rows).to_csv(s.reports_dir / "aasist_finetune_history.csv", index=False)
        if mean >= best["mean"]:
            return
        best["mean"] = mean
        labels = np.concatenate([val_sets[k][1] for k in val_scores])
        pooled = np.concatenate([val_scores[k] for k in val_scores])
        weights = folder / am.FINETUNED_FILE
        tmp = weights.with_suffix(".tmp")
        torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, tmp)
        tmp.replace(weights)
        meta = {
            "sha256": am.sha256_file(weights),
            "base_weights": f"clovaai/aasist@{am.COMMIT} (published AASIST.pth)",
            "data_source": f"ASVspoof 2019 LA train + In-the-Wild train ({n_train_speakers} "
            "speakers), fine-tuned from the published AASIST",
            "input": "raw",
            "threshold": float(compute_eer(labels, pooled)[1]),
            "val_eer": mean,
            "val_eer_by_set": r.eer,
            "val_eer_published_model": base,
            "epoch": r.epoch,
            "recipe": vars(args) | {"augmentation": "silence trim/pad, 8-12 kHz band-limit, "
                                    "mu-law, gain, noise", "device": describe_device(device)},
            "trained_at": datetime.now(UTC).isoformat(),
        }  # fmt: skip
        (folder / am.FINETUNED_META).write_text(json.dumps(meta, indent=2))
        print(f"[save] epoch {r.epoch}: mean val EER {mean:.4f}, threshold "
              f"{meta['threshold']:.3f} -> {weights}", flush=True)  # fmt: skip

    train(model, pools, val_sets, device, args.epochs, args.steps, args.batch, args.lr,
          args.seed, on_epoch=on_epoch, workers=args.workers, accum=args.accum)  # fmt: skip
    if not np.isfinite(best["mean"]):
        print("[warn] no epoch beat itself? nothing saved")
        return 1
    print(f"[ok  ] best mean validation EER {best['mean']:.4f} "
          f"(published model: {rows[0]['mean_eer']:.4f})")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())

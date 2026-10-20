"""Measure the audio detectors on ASVspoof 2019 LA dev and eval, overall and per attack.

Usage (from the repo root, after import_asvspoof.py and make_splits.py --dataset asvspoof):
    uv run python scripts/evaluate_audio.py --max-per-class 1500     # ~25 min on an M1
    uv run python scripts/evaluate_audio.py                          # everything (hours)

Variants, each fed exactly as the app feeds it (AudioExtractor → detector):
  aasist_raw       pretrained AASIST on the untrimmed waveform (how it was trained)
  aasist_trimmed   the same network on the silence-trimmed, peak-normalised waveform
                   (immune to the ASVspoof silence shortcut, Müller et al. 2021)
  aasist_ft        AASIST fine-tuned on In-the-Wild + ASVspoof (train_aasist.py), if present
  classic          our MFCC-statistics classifier (if train_audio.py has been run)

Thresholds come from **dev** only (EER point) and are applied unchanged to eval. Writes
reports/audio_eval.csv (overall), reports/audio_eval_per_attack.csv (EER per attack on
eval: A07-A19, A16/A19 reuse training-set algorithms) and models/audio/aasist/
calibration.json (the dev threshold of the chosen input, used by the app).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from deepsecure.config.device import describe_device, get_device
from deepsecure.config.settings import Settings, get_settings
from deepsecure.exceptions import DeepSecureError
from deepsecure.models.aasist_model import AASIST
from deepsecure.models.audio_model import AudioDetector
from deepsecure.models.metrics import binary_metrics, compute_eer
from deepsecure.preprocessing.audio_extractor import AudioExtractor


def _features(path: str, s: Settings):
    try:
        a = AudioExtractor(s).extract(path)
    except DeepSecureError:
        return None
    return a.raw_waveform, a.waveform, a.summary_vector


def score_split(
    df: pd.DataFrame,
    s: Settings,
    aasist: AASIST,
    classic,
    n_jobs: int,
    batch: int = 8,
    tuned: AASIST | None = None,
) -> pd.DataFrame:
    rows = []
    for start in range(0, len(df), 256):  # extract in chunks to bound memory (8 GB RAM)
        part = df.iloc[start : start + 256]
        feats = Parallel(n_jobs=n_jobs)(
            delayed(_features)(str(s.data_dir / p), s) for p in part["path"]
        )
        keep = [(r, f) for r, f in zip(part.itertuples(index=False), feats, strict=True) if f]
        for i in range(0, len(keep), batch):
            chunk = keep[i : i + batch]
            raw = aasist.batch_probs([f[0] for _, f in chunk])
            trimmed = aasist.batch_probs([f[1] for _, f in chunk])
            ft = tuned.batch_probs([f[0] for _, f in chunk]) if tuned else [np.nan] * len(chunk)
            cls = (
                classic.predict_proba(np.stack([f[2] for _, f in chunk]).astype(np.float64))[:, 1]
                if classic is not None
                else [np.nan] * len(chunk)
            )
            for (r, _), a, b, c, f in zip(chunk, raw, trimmed, cls, ft, strict=True):
                rows.append(
                    {
                        "utterance": r.utterance,
                        "attack": r.attack,
                        "label": int(r.label),
                        "aasist_raw": float(a),
                        "aasist_trimmed": float(b),
                        "classic": float(c),
                        "aasist_ft": float(f),
                    }
                )
        del feats, keep
        gc.collect()
        if aasist.device.type == "mps":
            torch.mps.empty_cache()  # give cached GPU buffers back between chunks
        print(f"[info] {min(start + 256, len(df))}/{len(df)} files", flush=True)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-per-class", type=int, default=None, help="per split (stratified)")
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--batch", type=int, default=8, help="AASIST batch (memory-bound)")
    parser.add_argument("--seed", type=int, default=s.split_seed)
    args = parser.parse_args(argv)

    csv = s.data_dir / "processed" / "splits" / "asvspoof2019_la.csv"
    if not csv.is_file():
        print("[error] run import_asvspoof.py and make_splits.py --dataset asvspoof first")
        return 1
    df = pd.read_csv(csv)
    df = df[df["exists"].astype(bool)] if "exists" in df else df
    device = get_device(s.device)
    det = AudioDetector(
        s.model_copy(update={"audio_backend": "aasist", "aasist_variant": "pretrained"}),
        device=device,
    )
    det.load()
    ft_det = AudioDetector(
        s.model_copy(update={"audio_backend": "aasist", "aasist_variant": "finetuned"}),
        device=device,
    )
    ft_det.load()
    tuned = ft_det.aasist if ft_det.info.get("model_name") == "AASIST (fine-tuned)" else None
    if det.aasist is None:
        print("[error] AASIST weights missing: uv run python scripts/download_assets.py")
        return 1
    classic_det = AudioDetector(s.model_copy(update={"audio_backend": "classic"}))
    classic = (
        classic_det.model if classic_det.load_classic(classic_det.default_checkpoint) else None
    )
    print(
        f"[info] device {describe_device(device)}; classic model: {classic is not None}; "
        f"fine-tuned AASIST: {tuned is not None}"
    )

    scored = {}
    for split in ("val", "test"):
        part = df[df["split"] == split]
        if args.max_per_class:
            # stratify spoofs by attack so every attack type is represented
            real = part[part["label"] == 0].sample(
                min(args.max_per_class, (part["label"] == 0).sum()), random_state=args.seed
            )
            spoof = part[part["label"] == 1]
            per_attack = max(1, args.max_per_class // spoof["attack"].nunique())
            spoof = (
                spoof.sample(frac=1.0, random_state=args.seed).groupby("attack").head(per_attack)
            )
            part = pd.concat([real, spoof])
        t = time.perf_counter()
        scored[split] = score_split(part, s, det.aasist, classic, args.n_jobs, args.batch, tuned)
        scored[split].to_csv(s.reports_dir / f"audio_scores_{split}.csv", index=False)
        print(f"[info] {split}: {len(scored[split])} files in {time.perf_counter() - t:.0f} s")

    rows, attack_rows = [], []
    variants = (
        ["aasist_raw", "aasist_trimmed"]
        + (["aasist_ft"] if tuned is not None else [])
        + (["classic"] if classic is not None else [])
    )
    thresholds = {}
    for v in variants:
        dev = scored["val"]
        thresholds[v] = compute_eer(dev["label"].to_numpy(), dev[v].to_numpy())[1]
        for split, name in (("val", "dev"), ("test", "eval")):
            d = scored[split]
            m = binary_metrics(d["label"].to_numpy(), d[v].to_numpy(), thresholds[v])
            rows.append(
                {
                    "variant": v,
                    "split": name,
                    "n": m["n"],
                    "n_spoof": m["n_fake"],
                    "eer": m["eer"],
                    "roc_auc": m["roc_auc"],
                    "threshold (dev EER)": thresholds[v],
                    "accuracy": m["accuracy"],
                    "recall_spoof": m["recall"],
                    "false_alarm_bonafide": m["fp"] / max(1, m["fp"] + m["tn"]),
                }
            )
        ev = scored["test"]
        bona = ev[ev["label"] == 0]
        for attack, grp in ev[ev["label"] == 1].groupby("attack"):
            both = pd.concat([bona, grp])
            attack_rows.append(
                {
                    "variant": v,
                    "attack": attack,
                    "n_spoof": len(grp),
                    "eer": compute_eer(both["label"].to_numpy(), both[v].to_numpy())[0],
                }
            )
    table = pd.DataFrame(rows)
    per_attack = pd.DataFrame(attack_rows).pivot(index="attack", columns="variant", values="eer")
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(s.reports_dir / "audio_eval.csv", index=False)
    per_attack.to_csv(s.reports_dir / "audio_eval_per_attack.csv")
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string(index=False))
        print("\nEER per eval attack:\n" + per_attack.round(4).to_string())

    chosen = s.aasist_input
    eer_of = table.set_index(["variant", "split"])["eer"]
    meta = {
        "input": chosen,
        "dev_eer_threshold": thresholds[f"aasist_{chosen}"],
        "dev_eer": float(eer_of[(f"aasist_{chosen}", "dev")]),
        "eval_eer": float(eer_of[(f"aasist_{chosen}", "eval")]),
        "n_dev": len(scored["val"]),
        "n_eval": len(scored["test"]),
        "measured_at": datetime.now(UTC).isoformat(),
        "device": describe_device(device),
    }
    path = det.aasist_weights.parent / "calibration.json"
    path.write_text(json.dumps(meta, indent=2))
    print(f"[ok  ] reports/audio_eval.csv, audio_eval_per_attack.csv; {path} ({chosen} input)")
    return 0


if __name__ == "__main__":
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    sys.exit(main())

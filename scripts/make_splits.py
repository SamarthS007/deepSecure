"""Build leakage-safe train/val/test split CSVs in data/processed/splits/.

Usage (from the repo root):
    uv run python scripts/make_splits.py                       # every dataset that is present
    uv run python scripts/make_splits.py --dataset faceforensics --strategy official
    uv run python scripts/make_splits.py --dataset asvspoof --max-per-class 2000

Writes one CSV per dataset (faceforensics.csv, asvspoof2019_la.csv, internal.csv)
with a `split` column, prints the real/fake balance per split, and refuses to
write any split where an identity group appears in more than one split.
Paths in the CSVs are relative to data/, so they also work on Colab.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from deepsecure.config.settings import get_settings
from deepsecure.data.asvspoof import load_asvspoof_la
from deepsecure.data.faceforensics import (
    apply_official_splits,
    load_official_splits,
    scan_faceforensics,
)
from deepsecure.data.internal import scan_internal
from deepsecure.data.splits import find_leakage, group_split, summarize, write_split_csv

DOCS_HINT = "see docs/DOCUMENTATION.md §8 for how to get it"


def build_faceforensics(args: argparse.Namespace, data_dir: Path) -> pd.DataFrame | None:
    root: Path = args.ff_root
    if not (root / "original_sequences").is_dir():
        print(f"[skip] FaceForensics++ not found at {root} ({DOCS_HINT})")
        return None
    df = scan_faceforensics(root, compression=args.compression, base=data_dir)
    if df.empty:
        print(f"[skip] no {args.compression} videos under {root}")
        return None
    if args.strategy == "official":
        df = apply_official_splits(df, load_official_splits(root / "splits"))
        if df.attrs.get("dropped"):
            print(f"[info] dropped {df.attrs['dropped']} videos not covered by official splits")
    else:
        df = group_split(df, ratios=tuple(args.ratios), seed=args.seed)
    df["ids"] = df["ids"].map("_".join)  # list -> plain text for the CSV
    return df


def build_asvspoof(args: argparse.Namespace, data_dir: Path) -> pd.DataFrame | None:
    root: Path = args.asv_root
    if not (root / "ASVspoof2019_LA_cm_protocols").is_dir():
        print(f"[skip] ASVspoof 2019 LA not found at {root} ({DOCS_HINT})")
        return None
    df = load_asvspoof_la(root, base=data_dir, max_per_class=args.max_per_class, seed=args.seed)
    missing = int((~df["exists"]).sum())
    if missing:
        print(f"[warn] {missing} audio files listed in the protocols are missing on disk")
    return df


def build_internal(args: argparse.Namespace, data_dir: Path) -> pd.DataFrame | None:
    df = scan_internal(args.internal_root, base=data_dir)
    if df.empty:
        print(f"[skip] internal test set is empty ({args.internal_root}/real|fake)")
        return None
    return df


BUILDERS = {
    "faceforensics": ("faceforensics.csv", build_faceforensics),
    "asvspoof": ("asvspoof2019_la.csv", build_asvspoof),
    "internal": ("internal.csv", build_internal),
}


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    raw = s.data_dir / "raw"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=[*BUILDERS, "all"], default="all")
    parser.add_argument("--ff-root", type=Path, default=raw / "faceforensics")
    parser.add_argument("--asv-root", type=Path, default=raw / "asvspoof2019" / "LA")
    parser.add_argument("--internal-root", type=Path, default=raw / "internal")
    parser.add_argument("--out", type=Path, default=s.data_dir / "processed" / "splits")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=s.data_dir,
        help="paths in the CSVs are stored relative to this folder",
    )
    parser.add_argument("--compression", default="c23", choices=["c0", "c23", "c40"])
    parser.add_argument(
        "--strategy",
        choices=["group", "official"],
        default="group",
        help="FaceForensics++: our 70/15/15 group split, or the official one",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=list(s.split_ratios),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument("--seed", type=int, default=s.split_seed)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="ASVspoof: cap bona fide/spoof files per split (quick experiments)",
    )
    args = parser.parse_args(argv)

    chosen = list(BUILDERS) if args.dataset == "all" else [args.dataset]
    written = 0
    for name in chosen:
        filename, build = BUILDERS[name]
        df = build(args, args.data_dir)
        if df is None:
            continue
        problems = find_leakage(df)
        if problems:
            print(f"[FAIL] {name}: leakage detected:\n  " + "\n  ".join(problems[:10]))
            return 1
        path = write_split_csv(df, args.out / filename)
        print(f"\n[ok  ] {name}: {len(df)} rows -> {path}")
        print(summarize(df).to_string())
        print("       leakage check: no group or file appears in more than one split")
        written += 1

    if written == 0:
        print("\nNo dataset found, so nothing was written. Download one first (docs §8).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

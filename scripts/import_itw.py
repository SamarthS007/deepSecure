"""Unpack In-the-Wild (Müller et al., Interspeech 2022) and split it by speaker.

Usage (from the repo root, after ``bash scripts/fetch_datasets.sh itw``):
    uv run python scripts/import_itw.py --delete-zip

In-the-Wild has real and deepfake speech of 58 celebrities and politicians, collected from
online videos: both classes come from the same kind of source (compressed internet audio),
so recording conditions don't give the label away, unlike ASVspoof (studio speech) or the
modern-TTS test set (YouTube real vs clean TTS fakes).

* ``meta.csv`` lists ``file, speaker, label`` (``bona-fide`` / ``spoof``); every row must
  match a WAV in the zip, and only plain file names are extracted.
* Splits are by **speaker** (``group``), 70/15/15 of the files, re-checked for leakage: a
  voice heard in training never appears in validation or test.

Writes data/raw/in_the_wild/wav/*.wav and data/processed/splits/in_the_wild.csv.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

from deepsecure.config.settings import get_settings
from deepsecure.data.splits import find_leakage, group_split, summarize

LABELS = {"bona-fide": 0, "spoof": 1}


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--delete-zip", action="store_true", help="free 8 GB after extracting")
    parser.add_argument("--seed", type=int, default=s.split_seed)
    args = parser.parse_args(argv)

    root = s.data_dir / "raw" / "in_the_wild"
    archive, wav_dir = root / "release_in_the_wild.zip", root / "wav"
    meta_path = root / "meta.csv"
    if not meta_path.is_file():
        if not archive.is_file():
            print(f"[error] {archive} not found: bash scripts/fetch_datasets.sh itw")
            return 1
        wav_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            members = {Path(m).name: m for m in z.namelist() if not m.endswith("/")}
            if "meta.csv" not in members:
                print("[error] meta.csv missing from the archive")
                return 1
            meta = pd.read_csv(io.BytesIO(z.read(members["meta.csv"])))
            missing = [f for f in meta["file"] if Path(f).name not in members]
            if missing or set(meta["label"]) - set(LABELS):
                print(f"[FAIL] meta.csv rows without audio ({len(missing)}) or unknown labels")
                return 1
            for i, name in enumerate(meta["file"].map(lambda f: Path(f).name), 1):
                target = wav_dir / name
                if not target.is_file():
                    target.write_bytes(z.read(members[name]))
                if i % 5000 == 0:
                    print(f"[info] extracted {i}/{len(meta)}", flush=True)
        meta.to_csv(meta_path, index=False)  # written last: marks a complete extraction
    meta = pd.read_csv(meta_path)
    if args.delete_zip and archive.is_file():
        archive.unlink()
        print(f"[info] deleted {archive.name}")

    df = pd.DataFrame(
        {
            "path": [f"raw/in_the_wild/wav/{Path(f).name}" for f in meta["file"]],
            "label": meta["label"].map(LABELS),
            "group": meta["speaker"],
            "attack": meta["label"].map({"bona-fide": "-", "spoof": "in-the-wild deepfake"}),
        }
    )
    df = group_split(df, s.split_ratios, args.seed).assign(dataset="in_the_wild")
    problems = find_leakage(df)
    if problems:
        print("[error] leakage:", problems[:5])
        return 1
    out = s.data_dir / "processed" / "splits" / "in_the_wild.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(summarize(df).to_string())
    print(
        f"[ok  ] {out}: {len(df)} clips, {df['group'].nunique()} speakers "
        f"({df.groupby('split')['group'].nunique().to_dict()} per split), no speaker in two splits"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

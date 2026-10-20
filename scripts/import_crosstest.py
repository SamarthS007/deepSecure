"""Unpack the two cross-dataset TEST sets (never used for training) and write their split CSVs.

Usage (from the repo root, after ``bash scripts/fetch_datasets.sh crosstest``):
    uv run python scripts/import_crosstest.py

DF40 test faces (data/raw/df40_test/test.zip, CC BY-NC 4.0)
    3,212 face images, 1,606 real / 1,606 fake from DF40's 40 newer methods. The re-upload
    has no per-method labels, so methods are reported as one pool. Real images are video
    frames named ``<video>_<frame>``; ``group`` keeps frames of one video together.
    -> data/processed/df40_test/{real,fake}/<name>, data/processed/splits/df40_test.csv

Modern TTS voices (data/raw/modern_tts/*.parquet, CC BY 4.0)
    1,866 FLAC clips: real speech from 14 YouTube recordings vs ElevenLabs (el), Amazon Polly
    (po), Speechify (sp), Hume AI (hu), Kokoro (hg) and Luvvoice (lv). The file-name prefix
    names the platform; each label is checked against it (yt = real, the rest = fake), and
    ``group`` is the source recording (e.g. yt_0009).
    -> data/processed/modern_tts/{real,fake}/<name>.flac, data/processed/splits/modern_tts.csv

Only plain file names are written (no paths from the archive are trusted).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from deepsecure.config.settings import get_settings

PLATFORMS = {
    "yt": "real (YouTube)",
    "el": "ElevenLabs",
    "po": "Amazon Polly",
    "sp": "Speechify",
    "hu": "Hume AI",
    "hg": "Kokoro",
    "lv": "Luvvoice",
}
IMAGE_EXT = (".png", ".jpg", ".jpeg")


def import_df40(raw: Path, out: Path) -> pd.DataFrame | None:
    archive = raw / "df40_test" / "test.zip"
    if not archive.is_file():
        print(f"[skip] {archive} not found: bash scripts/fetch_datasets.sh crosstest")
        return None
    rows = []
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            parts = member.split("/")
            if (
                len(parts) != 3
                or parts[1] not in ("real", "fake")
                or not parts[2].lower().endswith(IMAGE_EXT)
            ):
                continue
            cls, name = parts[1], Path(parts[2]).name  # basename only
            target = out / "df40_test" / cls / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                target.write_bytes(z.read(member))
            rows.append(
                {
                    "path": str(target.relative_to(out.parent)),
                    "label": int(cls == "fake"),
                    "attack": "DF40 (40 methods, unlabelled)" if cls == "fake" else "-",
                    "group": f"{cls}_{name.split('_')[0].split('.')[0]}",
                }
            )
    return pd.DataFrame(rows)


def import_modern_tts(raw: Path, out: Path) -> pd.DataFrame | None:
    files = sorted((raw / "modern_tts").glob("*.parquet"))
    if not files:
        print("[skip] data/raw/modern_tts/*.parquet not found: fetch_datasets.sh crosstest")
        return None
    rows, bad = [], []
    for f in files:
        for rec in pq.read_table(f).to_pylist():
            name = Path(rec["audio"]["path"]).name
            data = rec["audio"]["bytes"]
            prefix = name.split("_")[0]
            expected = 0 if prefix == "yt" else 1
            if prefix not in PLATFORMS or rec["label"] != expected or data[:4] != b"fLaC":
                bad.append(name)
                continue
            cls = "fake" if expected else "real"
            target = out / "modern_tts" / cls / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                target.write_bytes(data)
            rows.append(
                {
                    "path": str(target.relative_to(out.parent)),
                    "label": expected,
                    "attack": PLATFORMS[prefix],
                    "group": "_".join(name.split("_")[:2]),
                }
            )
    if bad:
        print(f"[FAIL] {len(bad)} clips with an unknown prefix, a label that disagrees with it")
        print(f"       or non-FLAC bytes, e.g. {bad[:3]}")
        return None
    return pd.DataFrame(rows)


def main() -> int:
    s = get_settings()
    raw, processed = s.data_dir / "raw", s.data_dir / "processed"
    splits = processed / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    status = 0
    for name, fn in (("df40_test", import_df40), ("modern_tts", import_modern_tts)):
        df = fn(raw, processed)
        if df is None:
            status |= int(name == "modern_tts" and any((raw / "modern_tts").glob("*.parquet")))
            continue
        df = df.assign(split="test", dataset=name).sort_values("path").reset_index(drop=True)
        df.to_csv(splits / f"{name}.csv", index=False)
        counts = df.groupby("attack")["path"].count().to_dict()
        print(f"[ok  ] {name}: {len(df)} files, {df['group'].nunique()} groups; {counts}")
    return status


if __name__ == "__main__":
    sys.exit(main())

"""Unpack the Hugging Face copy of ASVspoof 2019 LA into the official folder layout, verified.

Usage (from the repo root, after `bash scripts/fetch_datasets.sh asvspoof`):
    uv run python scripts/import_asvspoof.py
    uv run python scripts/import_asvspoof.py --delete-parquet     # free 7.5 GB afterwards

The copy (Hugging Face "Bisher/ASVspoof_2019_LA") stores each utterance's original FLAC
bytes with its speaker, file name, attack id and label. This script refuses to finish
unless, for every partition, the rows match the *official* CM protocol file exactly
(same utterances, speakers, attack ids and bona fide/spoof keys) and every audio blob is
a FLAC file. It writes:

    data/raw/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/*.txt
    data/raw/asvspoof2019/LA/ASVspoof2019_LA_{train,dev,eval}/flac/<utterance>.flac

which is exactly the layout of the official LA.zip, so `make_splits.py` works unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from deepsecure.config.settings import get_settings

PARTS = {  # parquet split -> (official partition folder, protocol file)
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    "validation": ("ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
    "test": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt"),
}


def read_protocol(path: Path) -> dict[str, tuple[str, str, str]]:
    """utterance -> (speaker, attack id, key) from an official CM protocol file."""
    out = {}
    for line in path.read_text().splitlines():
        speaker, utt, _, attack, key = line.split()
        out[utt] = (speaker, attack, key)
    return out


def import_split(parquet: Path, protocol: dict, flac_dir: Path) -> tuple[int, list[str]]:
    flac_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    seen = 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            utt = row["audio_file_name"].removesuffix(".flac")
            key = "bonafide" if row["key"] == 0 else "spoof"
            expected = protocol.get(utt)
            if expected is None:
                problems.append(f"{utt}: not in the official protocol")
            elif expected != (row["speaker_id"], row["system_id"], key):
                problems.append(
                    f"{utt}: {(row['speaker_id'], row['system_id'], key)} != {expected}"
                )
            data = row["audio"]["bytes"]
            if data[:4] != b"fLaC":
                problems.append(f"{utt}: audio is not FLAC")
            target = flac_dir / f"{utt}.flac"
            if not target.is_file() or target.stat().st_size != len(data):
                target.write_bytes(data)
            seen += 1
    missing = len(set(protocol)) - seen
    if missing:
        problems.append(f"{missing} protocol utterances missing from the copy")
    return seen, problems


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = s.data_dir / "raw" / "asvspoof2019"
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--delete-parquet", action="store_true")
    args = parser.parse_args(argv)

    la = args.root / "LA"
    proto_dir = la / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for split, (folder, proto_name) in PARTS.items():
        parquet = args.root / "parquet" / f"{split}.parquet"
        proto_src = args.root / "protocols" / proto_name
        if not parquet.is_file() or not proto_src.is_file():
            print(f"[skip] {split}: {parquet.name} or {proto_name} not downloaded yet")
            failed = True
            continue
        shutil.copy(proto_src, proto_dir / proto_name)
        protocol = read_protocol(proto_src)
        try:
            n, problems = import_split(parquet, protocol, la / folder / "flac")
        except pa.ArrowInvalid:
            print(f"[skip] {split}: {parquet.name} is incomplete (still downloading?)")
            failed = True
            continue
        if problems:
            failed = True
            print(f"[FAIL] {split}: {len(problems)} problems, e.g. {problems[:3]}")
        else:
            print(
                f"[ok  ] {split}: {n} utterances match the official protocol exactly "
                f"({sum(v[2] == 'bonafide' for v in protocol.values())} bona fide) "
                f"-> {la / folder}"
            )
    if failed:
        return 1
    if args.delete_parquet:
        shutil.rmtree(args.root / "parquet")
        print("[ok  ] deleted the parquet copies")
    print("[ok  ] next: uv run python scripts/make_splits.py --dataset asvspoof")
    return 0


if __name__ == "__main__":
    sys.exit(main())

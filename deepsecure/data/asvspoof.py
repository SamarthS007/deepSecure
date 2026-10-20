"""Read the ASVspoof 2019 Logical Access (LA) protocol files.

Layout after unzipping ``LA.zip`` into ``data/raw/asvspoof2019/``::

    LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
    LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
    LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
    LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac
    LA/ASVspoof2019_LA_dev/flac/...
    LA/ASVspoof2019_LA_eval/flac/...

Each protocol line is space-separated: speaker id, utterance id, an unused
column, the system (attack) id (``-`` for genuine speech, ``A01``...``A19`` for
spoofs), and the key (``bonafide`` or ``spoof``), e.g.::

    LA_0079 LA_T_1138215 - - bonafide

We keep the **official** partitions: train -> train, dev -> val, eval -> test.
They are speaker-disjoint, and eval contains attack algorithms that never appear
in train/dev, which is exactly the "new, unseen deepfake tool" situation a
real deployment faces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from deepsecure.data.splits import relative_to

PARTITIONS: dict[str, tuple[str, str]] = {
    # our split: (protocol file, audio folder)
    "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
    "val": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
    "test": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
}
PROTOCOL_DIR = "ASVspoof2019_LA_cm_protocols"


@dataclass(frozen=True)
class ProtocolEntry:
    """One line of a CM protocol file."""

    speaker: str
    utterance: str
    attack: str  # "-" for bona fide speech
    key: str  # "bonafide" or "spoof"


def parse_protocol_line(line: str) -> ProtocolEntry:
    """Parse one protocol line; tolerant of the unused 3rd column being absent."""
    parts = line.split()
    if len(parts) < 4 or parts[-1] not in {"bonafide", "spoof"}:
        raise ValueError(f"not an ASVspoof CM protocol line: {line!r}")
    return ProtocolEntry(speaker=parts[0], utterance=parts[1], attack=parts[-2], key=parts[-1])


def load_asvspoof_la(
    la_root: Path,
    base: Path | None = None,
    max_per_class: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the split table from the official LA protocols.

    Args:
        la_root: The ``LA`` folder.
        base: Folder that stored paths are made relative to (default: ``la_root``'s parent).
        max_per_class: If set, keep at most this many bona fide and this many spoof
            files *per split* (random, seeded). Useful for quick experiments on a laptop.
        seed: Seed for that subsampling.

    Returns:
        DataFrame with ``path, label, split, group, attack, utterance, exists, dataset``.
        ``group`` is the speaker id: the official partitions never share speakers.
    """
    la_root = Path(la_root)
    base = base or la_root.parent
    frames = []
    for split, (protocol_name, audio_dir) in PARTITIONS.items():
        protocol = la_root / PROTOCOL_DIR / protocol_name
        if not protocol.is_file():
            raise FileNotFoundError(f"ASVspoof protocol not found: {protocol}")
        rows = []
        for line in protocol.read_text().splitlines():
            if not line.strip():
                continue
            entry = parse_protocol_line(line)
            audio = la_root / audio_dir / "flac" / f"{entry.utterance}.flac"
            rows.append(
                {
                    "path": relative_to(audio, base),
                    "label": int(entry.key == "spoof"),
                    "split": split,
                    "group": entry.speaker,
                    "attack": entry.attack,
                    "utterance": entry.utterance,
                    "exists": audio.is_file(),
                    "dataset": "asvspoof2019_la",
                }
            )
        part = pd.DataFrame(rows)
        if max_per_class is not None and len(part):
            # Sample each class separately (stratified), so the subset keeps both classes.
            per_class = [
                part[part["label"] == lbl].sample(
                    n=min(int((part["label"] == lbl).sum()), max_per_class), random_state=seed
                )
                for lbl in (0, 1)
            ]
            part = pd.concat(per_class).sort_values("utterance")
        frames.append(part)
    return pd.concat(frames, ignore_index=True)

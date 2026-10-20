"""Scan a FaceForensics++ download and build identity groups.

Expected layout (what the official download script creates)::

    <root>/original_sequences/youtube/c23/videos/000.mp4
    <root>/manipulated_sequences/Deepfakes/c23/videos/000_003.mp4
    <root>/manipulated_sequences/Face2Face/c23/videos/...
    <root>/manipulated_sequences/FaceSwap/c23/videos/...
    <root>/manipulated_sequences/NeuralTextures/c23/videos/...

Naming: an original is ``<id>.mp4``. A fake is ``<target>_<source>.mp4``,
meaning "video <target> with the face/expressions of <source>". So a fake
contains *two* people, and both of their originals must be in the same split as
the fake; otherwise the test set shows faces the model was trained on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from deepsecure.data.splits import SPLITS, UnionFind, relative_to

METHODS: tuple[str, ...] = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
OFFICIAL_SPLITS_URL = (
    "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{split}.json"
)


def ids_in_name(stem: str) -> list[str]:
    """``"071"`` -> ``["071"]``; ``"071_054"`` -> ``["071", "054"]``."""
    parts = stem.split("_")
    if not parts or not all(p.isdigit() for p in parts) or len(parts) > 2:
        raise ValueError(f"not a FaceForensics++ video name: {stem!r}")
    return parts


def scan_faceforensics(
    root: Path,
    compression: str = "c23",
    methods: tuple[str, ...] = METHODS,
    base: Path | None = None,
) -> pd.DataFrame:
    """List every original and fake video under ``root`` for one compression level.

    Args:
        root: The FaceForensics++ download folder.
        compression: ``"c0"`` (raw), ``"c23"`` (high quality) or ``"c40"`` (low quality).
        methods: Manipulation methods to include.
        base: Folder that stored paths are made relative to (default: ``root``'s parent).

    Returns:
        DataFrame with ``path, label, method, video_id, ids, group`` columns.
    """
    root = Path(root)
    base = base or root.parent
    records: list[dict] = []

    originals = root / "original_sequences" / "youtube" / compression
    for path in sorted(originals.rglob("*.mp4")):
        records.append(_record(path, "original", 0, base))

    for method in methods:
        folder = root / "manipulated_sequences" / method / compression
        for path in sorted(folder.rglob("*.mp4")):
            records.append(_record(path, method, 1, base))

    df = pd.DataFrame(
        records, columns=["path", "label", "method", "video_id", "ids", "dataset"]
    )
    df["group"] = assign_groups(df["ids"]) if len(df) else pd.Series(dtype=str)
    return df


def assign_groups(ids_column: pd.Series) -> list[str]:
    """Give each row a group key so that linked identities share one key.

    ``000``, ``003``, ``000_003`` and ``003_017`` all end up in one group
    (``"ff_000"``): 000 is linked to 003 by the first fake, 003 to 017 by the second.
    """
    uf = UnionFind()
    for ids in ids_column:
        uf.union(ids)
    return [f"ff_{uf.find(ids[0])}" for ids in ids_column]


def load_official_splits(splits_dir: Path) -> dict[str, str]:
    """Map each video id to its official split from ``train/val/test.json``.

    The official files list *pairs* of ids, e.g. ``["071", "054"]``. Both ids of
    a pair, and both fakes ``071_054`` and ``054_071``, belong to that split.
    """
    id_to_split: dict[str, str] = {}
    for split in SPLITS:
        file = Path(splits_dir) / f"{split}.json"
        if not file.is_file():
            raise FileNotFoundError(
                f"{file} not found. Fetch the official splits with: "
                "uv run python scripts/download_assets.py --ff-splits"
            )
        for pair in json.loads(file.read_text()):
            for vid in pair:
                id_to_split[str(vid)] = split
    return id_to_split


def apply_official_splits(df: pd.DataFrame, id_to_split: dict[str, str]) -> pd.DataFrame:
    """Assign each video the official split of its ids.

    Rows whose ids aren't in the official lists, or whose two ids fall in
    different official splits, get ``split = None`` and are dropped with a count
    in ``df.attrs["dropped"]``, because guessing would risk leakage.
    """
    def split_for(ids: list[str]) -> str | None:
        splits = {id_to_split.get(i) for i in ids}
        return splits.pop() if len(splits) == 1 else None

    out = df.copy()
    out["split"] = out["ids"].map(split_for)
    dropped = int(out["split"].isna().sum())
    out = out.dropna(subset=["split"]).reset_index(drop=True)
    out.attrs["dropped"] = dropped
    return out


def _record(path: Path, method: str, label: int, base: Path) -> dict:
    ids = ids_in_name(path.stem)
    return {
        "path": relative_to(path, base),
        "label": label,
        "method": method,
        "video_id": path.stem,
        "ids": ids,
        "dataset": "faceforensics",
    }

"""Leakage-safe train/val/test splitting.

The golden rule: **everything that shares an identity must land in the same split.**
For video that means all frames of a video (and every fake made from that video's
person); for audio it means every utterance of a speaker. If the same person
appears in train and test, the model can score well by recognising the *person*
instead of detecting the *fake*: that's data leakage, and it inflates test accuracy.

Every dataset scanner in this package returns a DataFrame with at least these
columns, and the helpers below work on that shape:

    path   str  file path (relative to the data dir when possible)
    label  int  0 = real / bona fide, 1 = fake / spoof
    group  str  identity key; rows with the same group must share a split
    split  str  "train" | "val" | "test"  (added by a split function)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SPLITS: tuple[str, ...] = ("train", "val", "test")
REQUIRED_COLUMNS: tuple[str, ...] = ("path", "label", "group")


class UnionFind:
    """Merge items into connected groups ("if A is linked to B and B to C, all three are one group").

    Used for FaceForensics++, where a fake ``071_054.mp4`` links the people in
    originals ``071`` and ``054``; any chain of such links must stay together.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        """Return the representative ("root") of ``item``'s group."""
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # path compression: point straight at the root
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, items: Iterable[str]) -> None:
        """Put all ``items`` in the same group."""
        items = list(items)
        for other in items[1:]:
            a, b = self.find(items[0]), self.find(other)
            if a != b:
                self._parent[max(a, b)] = min(a, b)  # deterministic root choice


def group_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """Assign whole groups to train/val/test so that roughly ``ratios`` of the *rows* land in each.

    Groups are shuffled with a fixed seed (reproducible), then walked in order:
    each group goes to the split whose share is furthest below its target. That
    keeps the proportions close even when groups have different sizes.

    Args:
        df: DataFrame with ``path``, ``label`` and ``group`` columns.
        ratios: Target fractions for train, val and test (must sum to 1).
        seed: Random seed for the group shuffle.

    Returns:
        A copy of ``df`` with a ``split`` column.
    """
    _require_columns(df)
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0) or min(ratios) < 0:
        raise ValueError(f"ratios must be three non-negative numbers summing to 1, got {ratios}")

    sizes = df.groupby("group").size()
    rng = np.random.default_rng(seed)
    order = sizes.index.to_numpy().copy()
    rng.shuffle(order)

    total = int(sizes.sum())
    targets = np.asarray(ratios) * total
    filled = np.zeros(3)
    assignment: dict[str, str] = {}
    for group in order:
        deficit = targets - filled  # how far each split is from its target
        idx = int(np.argmax(deficit))
        assignment[group] = SPLITS[idx]
        filled[idx] += sizes[group]

    out = df.copy()
    out["split"] = out["group"].map(assignment)
    return out


def find_leakage(df: pd.DataFrame) -> list[str]:
    """Return human-readable leakage problems (an empty list means the split is clean).

    Checks that (1) no group appears in more than one split, and (2) no file path
    appears in more than one split.
    """
    _require_columns(df, extra=("split",))
    problems: list[str] = []

    groups_per_split = df.groupby("group")["split"].nunique()
    for group in groups_per_split[groups_per_split > 1].index:
        splits = sorted(df.loc[df["group"] == group, "split"].unique())
        problems.append(f"group {group!r} appears in several splits: {splits}")

    paths_per_split = df.groupby("path")["split"].nunique()
    for path in paths_per_split[paths_per_split > 1].index:
        problems.append(f"file {path!r} appears in several splits")

    bad_splits = set(df["split"].dropna().unique()) - set(SPLITS)
    if bad_splits:
        problems.append(f"unknown split names: {sorted(bad_splits)}")
    if df["split"].isna().any():
        problems.append(f"{int(df['split'].isna().sum())} rows have no split assigned")
    return problems


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Count real/fake rows and groups per split; shows class balance at a glance."""
    _require_columns(df, extra=("split",))
    rows = []
    for split in SPLITS:
        part = df[df["split"] == split]
        n_real = int((part["label"] == 0).sum())
        n_fake = int((part["label"] == 1).sum())
        n = n_real + n_fake
        rows.append(
            {
                "split": split,
                "real": n_real,
                "fake": n_fake,
                "total": n,
                "fake_ratio": round(n_fake / n, 3) if n else float("nan"),
                "groups": int(part["group"].nunique()),
                "share": round(n / len(df), 3) if len(df) else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("split")


def relative_to(path: Path, base: Path) -> str:
    """Store paths relative to the data folder so the CSVs also work on another machine (e.g. Colab)."""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_split_csv(df: pd.DataFrame, out_path: Path) -> Path:
    """Write a split table after re-checking it for leakage (refuses to write a leaky split)."""
    problems = find_leakage(df)
    if problems:
        raise ValueError("refusing to write a leaky split:\n  " + "\n  ".join(problems[:10]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)
    return out_path


def _require_columns(df: pd.DataFrame, extra: tuple[str, ...] = ()) -> None:
    missing = [c for c in (*REQUIRED_COLUMNS, *extra) if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

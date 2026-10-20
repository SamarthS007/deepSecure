"""The internal "scam-style" test set: your own labelled clips.

Layout::

    data/raw/internal/real/<any name>.mp4|.wav|...   genuine recordings (with consent)
    data/raw/internal/fake/<any name>.mp4|.wav|...   deepfakes / voice clones

The folder name *is* the label. Every file goes to the ``test`` split: this set
is never trained on. It measures how well the system copes with media unlike the
academic datasets. ``group`` defaults to the file name; if several clips show
the same person, set a shared ``person`` value in ``metadata.csv`` (see the
README there) so the group reflects that.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from deepsecure.config.settings import get_settings
from deepsecure.data.splits import relative_to

LABEL_FOLDERS: dict[str, int] = {"real": 0, "fake": 1}


def scan_internal(root: Path, base: Path | None = None) -> pd.DataFrame:
    """List the media files in ``root/real`` and ``root/fake``.

    Args:
        root: The internal test-set folder.
        base: Folder that stored paths are made relative to (default: ``root``'s parent).

    Returns:
        DataFrame with ``path, label, split, group, media_type, dataset``.
    """
    settings = get_settings()
    root = Path(root)
    base = base or root.parent
    media_types = {ext: "video" for ext in settings.video_extensions}
    media_types |= {ext: "audio" for ext in settings.audio_extensions}

    people: dict[str, str] = {}
    metadata = root / "metadata.csv"
    if metadata.is_file():
        meta = pd.read_csv(metadata, dtype=str).fillna("")
        if {"filename", "person"} <= set(meta.columns):
            people = {r.filename: r.person for r in meta.itertuples() if r.person}

    rows = []
    for folder, label in LABEL_FOLDERS.items():
        for path in sorted((root / folder).glob("*")):
            ext = path.suffix.lower()
            if not path.is_file() or ext not in media_types:
                continue
            rows.append(
                {
                    "path": relative_to(path, base),
                    "label": label,
                    "split": "test",
                    "group": "internal_" + people.get(path.name, path.stem),
                    "media_type": media_types[ext],
                    "dataset": "internal",
                }
            )
    columns = ["path", "label", "split", "group", "media_type", "dataset"]
    return pd.DataFrame(rows, columns=columns)

"""End-to-end run of scripts/make_splits.py on synthetic dataset folders."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_splits.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=ROOT
    )


def test_make_splits_writes_leak_free_csvs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    orig = data / "raw" / "faceforensics" / "original_sequences" / "youtube" / "c23" / "videos"
    orig.mkdir(parents=True)
    fake = data / "raw" / "faceforensics" / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    fake.mkdir(parents=True)
    for i in range(0, 40, 2):
        (orig / f"{i:03d}.mp4").touch()
        (orig / f"{i + 1:03d}.mp4").touch()
        (fake / f"{i:03d}_{i + 1:03d}.mp4").touch()
    (data / "raw" / "internal" / "real").mkdir(parents=True)
    (data / "raw" / "internal" / "real" / "me.mp4").touch()

    out = tmp_path / "splits"
    result = _run(
        "--data-dir", str(data),
        "--ff-root", str(data / "raw" / "faceforensics"),
        "--asv-root", str(data / "raw" / "missing"),
        "--internal-root", str(data / "raw" / "internal"),
        "--out", str(out),
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "[skip] ASVspoof 2019 LA not found" in result.stdout

    ff = pd.read_csv(out / "faceforensics.csv", dtype={"video_id": str})
    assert len(ff) == 60
    assert ff.groupby("group")["split"].nunique().max() == 1
    assert ff["path"].str.startswith("raw/faceforensics/").all()
    assert set(ff["split"]) == {"train", "val", "test"}

    internal = pd.read_csv(out / "internal.csv")
    assert internal["split"].tolist() == ["test"]


def test_make_splits_with_no_data_writes_nothing(tmp_path: Path) -> None:
    result = _run(
        "--data-dir", str(tmp_path),
        "--ff-root", str(tmp_path / "a"),
        "--asv-root", str(tmp_path / "b"),
        "--internal-root", str(tmp_path / "c"),
        "--out", str(tmp_path / "splits"),
    )  # fmt: skip
    assert result.returncode == 0
    assert "nothing was written" in result.stdout
    assert not (tmp_path / "splits").exists()

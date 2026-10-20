"""FaceForensics++, ASVspoof and internal-set scanners on synthetic folder trees.

The scanners only look at file *names*, so empty placeholder files are enough.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepsecure.data.asvspoof import PARTITIONS, PROTOCOL_DIR, load_asvspoof_la, parse_protocol_line
from deepsecure.data.faceforensics import (
    apply_official_splits,
    ids_in_name,
    load_official_splits,
    scan_faceforensics,
)
from deepsecure.data.internal import scan_internal
from deepsecure.data.splits import find_leakage, group_split

# --- FaceForensics++ -------------------------------------------------------


@pytest.fixture
def ff_root(tmp_path: Path) -> Path:
    root = tmp_path / "faceforensics"
    orig = root / "original_sequences" / "youtube" / "c23" / "videos"
    for vid in ["000", "003", "017", "100", "200", "201"]:
        (orig / f"{vid}.mp4").parent.mkdir(parents=True, exist_ok=True)
        (orig / f"{vid}.mp4").touch()
    (root / "original_sequences" / "youtube" / "c40" / "videos").mkdir(parents=True)
    (root / "original_sequences" / "youtube" / "c40" / "videos" / "000.mp4").touch()
    fakes = {"Deepfakes": ["000_003", "003_000"], "FaceSwap": ["003_017"], "Face2Face": ["200_201"]}
    for method, names in fakes.items():
        folder = root / "manipulated_sequences" / method / "c23" / "videos"
        folder.mkdir(parents=True)
        for name in names:
            (folder / f"{name}.mp4").touch()
    return root


def test_ids_in_name() -> None:
    assert ids_in_name("071") == ["071"]
    assert ids_in_name("071_054") == ["071", "054"]
    with pytest.raises(ValueError):
        ids_in_name("README")


def test_scan_faceforensics_labels_and_compression(ff_root: Path) -> None:
    df = scan_faceforensics(ff_root, compression="c23")
    assert len(df) == 6 + 4  # 6 originals + 4 fakes; the c40 file is ignored
    assert set(df.loc[df["method"] == "original", "label"]) == {0}
    assert set(df.loc[df["method"] != "original", "label"]) == {1}
    assert df["path"].str.startswith("faceforensics/").all()  # relative to the parent folder


def test_faceforensics_groups_link_people_across_fakes(ff_root: Path) -> None:
    df = scan_faceforensics(ff_root).set_index("video_id")
    # 000 <-> 003 (fake 000_003) and 003 <-> 017 (fake 003_017): one identity group.
    linked = {"000", "003", "017", "000_003", "003_000", "003_017"}
    assert df.loc[sorted(linked), "group"].nunique() == 1
    assert df.loc["100", "group"] != df.loc["000", "group"]
    assert df.loc["200", "group"] == df.loc["201", "group"] == df.loc["200_201", "group"]


def test_faceforensics_group_split_has_no_leakage(ff_root: Path) -> None:
    df = group_split(scan_faceforensics(ff_root), seed=3)
    assert find_leakage(df) == []


def test_official_splits(ff_root: Path, tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.json").write_text(json.dumps([["000", "003"], ["017", "100"]]))
    (splits / "val.json").write_text(json.dumps([["200", "201"]]))
    (splits / "test.json").write_text(json.dumps([]))
    df = apply_official_splits(scan_faceforensics(ff_root), load_official_splits(splits))
    by_id = df.set_index("video_id")["split"]
    assert by_id["000_003"] == "train"
    assert by_id["200_201"] == "val"
    # 003_017: both ids are in train -> kept.
    assert by_id["003_017"] == "train"
    assert df.attrs["dropped"] == 0


def test_official_splits_missing_files_give_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--ff-splits"):
        load_official_splits(tmp_path)


# --- ASVspoof 2019 LA ------------------------------------------------------


PROTOCOL_LINES = {
    "train": ["LA_0079 LA_T_0000001 - - bonafide", "LA_0079 LA_T_0000002 - A01 spoof",
              "LA_0080 LA_T_0000003 - A02 spoof"],
    "val": ["LA_0069 LA_D_0000001 - - bonafide", "LA_0069 LA_D_0000002 - A03 spoof"],
    "test": ["LA_0039 LA_E_0000001 - - bonafide", "LA_0039 LA_E_0000002 - A13 spoof",
             "LA_0040 LA_E_0000003 - A19 spoof"],
}  # fmt: skip


@pytest.fixture
def la_root(tmp_path: Path) -> Path:
    root = tmp_path / "asvspoof2019" / "LA"
    (root / PROTOCOL_DIR).mkdir(parents=True)
    for split, (protocol, audio_dir) in PARTITIONS.items():
        (root / PROTOCOL_DIR / protocol).write_text("\n".join(PROTOCOL_LINES[split]) + "\n")
        flac = root / audio_dir / "flac"
        flac.mkdir(parents=True)
        for line in PROTOCOL_LINES[split][:-1]:  # leave the last file missing on purpose
            (flac / f"{line.split()[1]}.flac").touch()
    return root


def test_parse_protocol_line() -> None:
    e = parse_protocol_line("LA_0079 LA_T_1138215 - - bonafide")
    assert (e.speaker, e.utterance, e.attack, e.key) == ("LA_0079", "LA_T_1138215", "-", "bonafide")
    assert parse_protocol_line("LA_0079 LA_T_1 - A05 spoof").attack == "A05"
    with pytest.raises(ValueError):
        parse_protocol_line("garbage line")


def test_load_asvspoof_maps_official_partitions(la_root: Path) -> None:
    df = load_asvspoof_la(la_root)
    assert df["split"].value_counts().to_dict() == {"train": 3, "test": 3, "val": 2}
    assert df.loc[df["attack"] == "-", "label"].eq(0).all()
    assert df.loc[df["attack"] != "-", "label"].eq(1).all()
    assert int((~df["exists"]).sum()) == 3  # one missing file per partition
    assert find_leakage(df) == []  # speakers are disjoint across partitions


def test_asvspoof_max_per_class_is_stratified(la_root: Path) -> None:
    df = load_asvspoof_la(la_root, max_per_class=1)
    for split in ("train", "val", "test"):
        assert sorted(df.loc[df["split"] == split, "label"]) == [0, 1]


# --- internal set ------------------------------------------------------------


def test_scan_internal(tmp_path: Path) -> None:
    root = tmp_path / "internal"
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir()
    (root / "real" / "me_01.mp4").touch()
    (root / "real" / "me_02.MOV").touch()  # extension check is case-insensitive
    (root / "fake" / "clone.wav").touch()
    (root / "fake" / "notes.txt").touch()  # not media: ignored
    (root / "real" / ".gitkeep").touch()
    (root / "metadata.csv").write_text("filename,person\nme_01.mp4,me\nme_02.MOV,me\n")

    df = scan_internal(root)
    assert len(df) == 3
    assert (df["split"] == "test").all()
    assert df.set_index("path").loc["internal/fake/clone.wav", "label"] == 1
    assert df.loc[df["label"] == 0, "group"].nunique() == 1  # both clips of "me" share a group
    assert set(df["media_type"]) == {"video", "audio"}

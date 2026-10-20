"""Group-aware splitting and leakage detection."""

from __future__ import annotations

import pandas as pd
import pytest

from deepsecure.data.splits import (
    UnionFind,
    find_leakage,
    group_split,
    summarize,
    write_split_csv,
)


def _frames(n_groups: int = 40, per_group: int = 5) -> pd.DataFrame:
    rows = [
        {"path": f"v{g}/f{i}.png", "label": g % 2, "group": f"v{g}"}
        for g in range(n_groups)
        for i in range(per_group)
    ]
    return pd.DataFrame(rows)


def test_union_find_merges_chains() -> None:
    uf = UnionFind()
    uf.union(["000", "003"])
    uf.union(["003", "017"])
    uf.union(["500"])
    assert uf.find("000") == uf.find("017") == "000"
    assert uf.find("500") != uf.find("000")


def test_group_split_keeps_groups_together() -> None:
    out = group_split(_frames(), seed=1)
    assert out.groupby("group")["split"].nunique().max() == 1
    assert find_leakage(out) == []


def test_group_split_hits_target_ratios() -> None:
    out = group_split(_frames(n_groups=100), ratios=(0.70, 0.15, 0.15), seed=0)
    shares = out["split"].value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.03)
    assert shares["val"] == pytest.approx(0.15, abs=0.03)
    assert shares["test"] == pytest.approx(0.15, abs=0.03)


def test_group_split_is_reproducible() -> None:
    a = group_split(_frames(), seed=7)["split"].tolist()
    b = group_split(_frames(), seed=7)["split"].tolist()
    c = group_split(_frames(), seed=8)["split"].tolist()
    assert a == b
    assert a != c


def test_group_split_rejects_bad_ratios() -> None:
    with pytest.raises(ValueError):
        group_split(_frames(), ratios=(0.5, 0.5, 0.5))


def test_find_leakage_catches_shared_group() -> None:
    df = group_split(_frames(), seed=0)
    first = df.index[0]
    other = "test" if df.loc[first, "split"] != "test" else "train"
    df.loc[first, "split"] = other  # one frame of a video jumps to another split
    problems = find_leakage(df)
    assert any("appears in several splits" in p for p in problems)


def test_write_split_csv_refuses_leaky_split(tmp_path) -> None:
    df = pd.DataFrame(
        {"path": ["a", "b"], "label": [0, 1], "group": ["g", "g"], "split": ["train", "test"]}
    )
    with pytest.raises(ValueError, match="leaky"):
        write_split_csv(df, tmp_path / "x.csv")
    assert not (tmp_path / "x.csv").exists()


def test_summarize_counts_classes() -> None:
    df = pd.DataFrame(
        {
            "path": list("abcd"),
            "label": [0, 1, 1, 1],
            "group": list("abcd"),
            "split": ["train", "train", "train", "test"],
        }
    )
    s = summarize(df)
    assert (s.loc["train", "real"], s.loc["train", "fake"]) == (1, 2)
    assert s.loc["train", "fake_ratio"] == pytest.approx(0.667, abs=1e-3)
    assert s.loc["val", "total"] == 0

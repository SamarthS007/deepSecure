"""Phase 9: system-level metrics, latency tables and the evaluate_system.py script."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deepsecure.config.settings import Settings
from deepsecure.models.system_eval import latency_table, method_metrics, run_files
from deepsecure.pipeline import DeepSecurePipeline
from deepsecure.utils.synthetic_media import write_corrupted_file, write_tone_wav
from tests.conftest import ROOT, isolated_models_dir, needs_ffmpeg, needs_mediapipe_assets


def rows() -> pd.DataFrame:
    """Two videos with all three scores, one photo (visual only), one rejected file."""
    return pd.DataFrame(
        [
            {
                "status": "ok",
                "label": 0,
                "media_type": "video",
                "duration_s": 4.0,
                "fused": 0.1,
                "visual": 0.2,
                "audio": 0.1,
                "lipsync": 0.1,
                "risk_level": "SAFE",
                "ms_validate": 10.0,
                "ms_total": 1000.0,
            },
            {
                "status": "ok",
                "label": 1,
                "media_type": "video",
                "duration_s": 4.0,
                "fused": 0.8,
                "visual": 0.7,
                "audio": 0.9,
                "lipsync": 0.8,
                "risk_level": "MALICIOUS",
                "ms_validate": 12.0,
                "ms_total": 1400.0,
            },
            {
                "status": "ok",
                "label": 1,
                "media_type": "image",
                "duration_s": 0.0,
                "fused": 0.3,
                "visual": 0.3,
                "audio": np.nan,
                "lipsync": np.nan,
                "risk_level": "SUSPICIOUS",
                "ms_validate": 8.0,
                "ms_total": 300.0,
            },
            {"status": "rejected: file is empty", "label": 1},
        ]
    )


def test_method_metrics_available_common_and_verdict() -> None:
    table = pd.DataFrame(method_metrics(rows(), "toy", threshold=0.4)).set_index(
        ["method", "subset"]
    )
    assert table.loc[("visual", "available"), "n"] == 3
    assert table.loc[("audio", "available"), "coverage"] == pytest.approx(2 / 3)
    assert table.loc[("fused", "common (all 3 ran)"), "n"] == 2
    assert table.loc[("fused", "available"), "threshold"] == 0.4
    verdict = table.loc[("verdict (not SAFE = flagged)", "all")]
    assert verdict["fakes_flagged"] == 1.0 and verdict["reals_flagged"] == 0.0
    assert verdict["fakes_malicious"] == 0.5
    flat = table.reset_index()
    assert flat.loc[flat["method"] == "rejected inputs", "n"].tolist() == [1]


def test_one_class_subset_is_reported_not_crashed() -> None:
    only_fakes = rows().iloc[[1, 2]]
    table = pd.DataFrame(method_metrics(only_fakes, "toy"))
    assert table["status"].str.contains("one class only").any()


def test_latency_table_and_real_time_factor() -> None:
    table = latency_table(rows(), "CPU", label="toy").set_index("stage")
    assert table.loc["validate", "n"] == 3 and table.loc["validate", "mean_ms"] == pytest.approx(
        10.0
    )
    rtf = table.loc["real-time factor (s per s of media)"]
    assert rtf["n"] == 2  # the photo has no duration and is excluded
    assert rtf["mean_ms"] == pytest.approx((1.0 / 4 + 1.4 / 4) / 2)
    assert (table["device"] == "CPU").all()


@needs_mediapipe_assets
def test_run_files_through_the_real_pipeline(tmp_path: Path) -> None:
    s = Settings(
        _env_file=None,
        visual_arch="test_resnet",
        device="cpu",
        models_dir=isolated_models_dir(tmp_path),
    )
    files = pd.DataFrame(
        {
            "path": [
                str(write_tone_wav(tmp_path / "v.wav", 2.0)),
                str(write_corrupted_file(tmp_path / "bad.mp4")),
            ],
            "label": [0, 1],
        }
    )
    pipe = DeepSecurePipeline(s)
    try:
        out = run_files(pipe, files, "toy").set_index("file")
    finally:
        pipe.close()
    assert out.loc["v.wav", "status"] == "ok" and np.isnan(out.loc["v.wav", "visual"])
    assert out.loc["v.wav", "ms_total"] > 0
    assert out.loc["bad.mp4", "status"].startswith("rejected")


def _script(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "DEEPSECURE_MODELS_DIR": str(isolated_models_dir(tmp_path)),
        "DEEPSECURE_REPORTS_DIR": str(tmp_path / "reports"),
        "DEEPSECURE_DATA_DIR": str(tmp_path / "data"),
        "DEEPSECURE_VISUAL_ARCH": "test_resnet",
        "DEEPSECURE_DEVICE": "cpu",
    }
    return subprocess.run(
        [sys.executable, "scripts/evaluate_system.py", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )


@needs_mediapipe_assets
def test_script_skips_untrained_detectors(tmp_path: Path) -> None:
    result = _script(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "TBD" in result.stdout
    table = pd.read_csv(tmp_path / "reports" / "system_eval.csv")
    assert table["status"].iloc[0].endswith("TBD")


@needs_mediapipe_assets
@needs_ffmpeg
@pytest.mark.slow
def test_script_benchmark(tmp_path: Path) -> None:
    result = _script(tmp_path, "--benchmark", "--durations", "2", "--repeats", "1")
    assert result.returncode == 0, result.stderr
    table = pd.read_csv(tmp_path / "reports" / "latency.csv")
    assert "real-time factor (s per s of media)" in set(table["stage"])
    assert (table["mean_ms"] > 0).all()


def test_open_sets_pick_heldout_files_per_class(tmp_path: Path) -> None:
    sys.argv = ["x"]
    from scripts.evaluate_system import open_sets

    (tmp_path / "raw" / "ravdess_test" / "Actor_07").mkdir(parents=True)
    (tmp_path / "raw" / "ravdess_test" / "Actor_07" / "a.mp4").write_bytes(b"")
    splits = tmp_path / "processed" / "splits"
    splits.mkdir(parents=True)
    pd.DataFrame(
        {
            "path": [f"img{i}.jpg" for i in range(8)],
            "label": [0, 1] * 4,
            "split": ["test"] * 6 + ["train"] * 2,
        }
    ).to_csv(splits / "deepfakeface.csv", index=False)
    sets = open_sets(tmp_path, 2, seed=0)
    assert set(sets) == {"ravdess_heldout", "deepfakeface_test"}  # ASVspoof not imported
    assert sets["ravdess_heldout"]["label"].tolist() == [0]
    dff = sets["deepfakeface_test"]
    assert dff["label"].value_counts().to_dict() == {0: 2, 1: 2}
    assert not dff["path"].str.contains("img6|img7").any()  # train files never used
    assert all(Path(p).is_absolute() for p in dff["path"])

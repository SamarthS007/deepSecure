"""Phase 4D: metrics, figures, score collection and the evaluation script."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from deepsecure.config.settings import Settings
from deepsecure.models.audio_model import AudioDetector
from deepsecure.models.evaluation import (
    EvalSet,
    eer_threshold,
    eval_sets_from_files,
    report_set,
    score_files,
    visual_video_scores,
)
from deepsecure.models.lipsync_model import LipSyncDetector
from deepsecure.models.metrics import binary_metrics, compute_eer
from deepsecure.models.visual_model import VisualDetector
from deepsecure.preprocessing.frame_extractor import FrameExtractor
from deepsecure.utils.synthetic_media import (
    synthetic_face_crops,
    write_corrupted_file,
    write_face_photo,
    write_talking_face_clip,
)
from tests.conftest import isolated_models_dir, needs_mediapipe_assets

ROOT = Path(__file__).resolve().parents[1]


# --- metrics ---------------------------------------------------------------------------


def test_binary_metrics_counts() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.6, 0.2, 0.9, 0.4, 0.8])  # one false positive, one false negative
    m = binary_metrics(y, p, threshold=0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["n"] == 6 and m["n_fake"] == 3


def test_binary_metrics_with_one_class() -> None:
    m = binary_metrics(np.zeros(4), np.array([0.1, 0.2, 0.7, 0.3]), 0.5)
    assert np.isnan(m["roc_auc"]) and np.isnan(m["eer"])
    assert m["fp"] == 1 and m["precision"] == 0.0  # no division-by-zero crash


def test_eer_threshold_sits_in_the_middle_of_the_gap() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    eer, thr = compute_eer(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]))
    assert eer == 0.0 and thr == pytest.approx(0.5)


def test_eer_threshold_falls_back_without_both_classes() -> None:
    assert eer_threshold(np.array([1, 1]), np.array([0.2, 0.9])) == 0.5
    assert eer_threshold(np.array([]), np.array([])) == 0.5


# --- reports ----------------------------------------------------------------------------


def _set(labels, scores, **kw) -> EvalSet:
    return EvalSet("audio", "toy", "test", np.array(labels), np.array(scores), 0.5, "0.5",
                   trained=True, **kw)  # fmt: skip


def test_report_set_writes_everything(tmp_path: Path) -> None:
    row = report_set(_set([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9]), tmp_path)
    assert row["status"] == "ok" and row["roc_auc"] == 1.0
    for name in ("roc", "confusion"):
        assert (tmp_path / f"{name}_audio_toy_test.png").stat().st_size > 10_000
    scores = pd.read_csv(tmp_path / "scores_audio_toy_test.csv")
    assert list(scores.columns) == ["id", "label", "score"] and len(scores) == 4


def test_report_set_one_class_and_empty(tmp_path: Path) -> None:
    row = report_set(_set([0, 0], [0.1, 0.2]), tmp_path)
    assert "one class only" in row["status"]
    assert not (tmp_path / "roc_audio_toy_test.png").exists()
    assert report_set(_set([], []), tmp_path)["status"].startswith("skipped")


def test_eval_sets_count_unavailable_items() -> None:
    scored = pd.DataFrame(
        {
            "path": ["a", "b", "c", "d"],
            "label": [0, 1, 1, 0],
            "status": ["ok"] * 3 + ["rejected"],
            "visual": [0.2, 0.9, np.nan, 0.1],
            "audio": [np.nan] * 4,
            "lipsync": [0.3, np.nan, 0.8, 0.5],
        }
    )
    thr = {m: (0.5, "0.5") for m in ("visual", "audio", "lipsync")}
    sets = {e.modality: e for e in eval_sets_from_files(
        scored, "internal", "test", thr, dict.fromkeys(thr, False), {})}  # fmt: skip
    assert len(sets["visual"].labels) == 2 and sets["visual"].n_unavailable == 1
    assert len(sets["audio"].labels) == 0 and sets["audio"].n_unavailable == 3
    assert sets["lipsync"].ids == ["a", "c"]  # the rejected file isn't counted at all


# --- score collection ---------------------------------------------------------------------


def test_visual_video_scores_group_by_video(tmp_path: Path) -> None:
    rows = []
    for v, label in (("v0", 0), ("v1", 1)):
        for i, crop in enumerate(synthetic_face_crops(3, bool(label), seed=len(rows))):
            Image.fromarray(crop).save(tmp_path / f"{v}_{i}.png")
            rows.append({"path": f"{v}_{i}.png", "label": label, "split": "test", "video": v})
    rows.append({"path": "v0_0.png", "label": 0, "split": "val", "video": "other"})
    det = VisualDetector(Settings(_env_file=None, visual_arch="test_resnet", device="cpu",
                                  models_dir=tmp_path), device=torch.device("cpu"))  # fmt: skip
    det.load()
    ids, labels, scores = visual_video_scores(det, pd.DataFrame(rows), "test", tmp_path)
    assert ids == ["v0", "v1"] and list(labels) == [0, 1]
    assert scores.shape == (2,) and ((scores >= 0) & (scores <= 1)).all()


@needs_mediapipe_assets
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_score_files_runs_the_app_stages(tmp_path: Path) -> None:
    s = Settings(_env_file=None, data_dir=tmp_path, visual_arch="test_resnet", device="cpu",
                 models_dir=isolated_models_dir(tmp_path))  # fmt: skip
    write_talking_face_clip(tmp_path / "talk.mp4", 3.0, rhythm="speech")
    write_face_photo(tmp_path / "photo.jpg")
    write_corrupted_file(tmp_path / "bad.mp4")
    files = pd.DataFrame({"path": ["talk.mp4", "photo.jpg", "bad.mp4"], "label": [0, 1, 1]})
    visual, audio, lipsync = (
        VisualDetector(s, torch.device("cpu")),
        AudioDetector(s),
        LipSyncDetector(s),
    )
    visual.load()
    with FrameExtractor(s) as fx:
        scored = score_files(files, s, visual, audio, lipsync, fx).set_index("path")
    assert scored.loc["bad.mp4", "status"].startswith("rejected")
    assert scored.loc["talk.mp4", "status"] == "ok"
    assert not np.isnan(scored.loc["talk.mp4", ["visual", "audio", "lipsync"]].astype(float)).any()
    assert np.isnan(scored.loc["photo.jpg", "lipsync"])  # a photo has no lip-sync
    assert "no audio" in scored.loc["photo.jpg", "audio_reason"]


# --- the script -----------------------------------------------------------------------------


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {**os.environ, "DEEPSECURE_DATA_DIR": str(tmp_path / "data"),
            "DEEPSECURE_MODELS_DIR": str(tmp_path / "models"),
            "DEEPSECURE_REPORTS_DIR": str(tmp_path / "reports"), **extra}  # fmt: skip


def test_script_skips_untrained_detectors(tmp_path: Path) -> None:
    """With nothing trained, nothing is measured: every row is a TBD placeholder."""
    result = subprocess.run([sys.executable, "scripts/evaluate_models.py"], capture_output=True,
                            text=True, env=_env(tmp_path), cwd=ROOT)  # fmt: skip
    assert result.returncode == 0, result.stderr
    table = pd.read_csv(tmp_path / "reports" / "modality_metrics.csv")
    assert len(table) == 5
    assert table["status"].str.contains("not trained").all()
    assert table["status"].str.endswith("TBD").all()
    assert "accuracy" not in table.columns


@needs_mediapipe_assets
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_script_on_an_internal_set(tmp_path: Path) -> None:
    """--allow-untrained on a 2-clip internal set: rows are measured but marked trained=False."""
    data = tmp_path / "data"
    write_talking_face_clip(data / "raw" / "internal" / "real" / "a.mp4", 3.0, rhythm="speech")
    write_talking_face_clip(data / "raw" / "internal" / "fake" / "b.mp4", 3.0, lag_s=0.5,
                            rhythm="speech")  # fmt: skip
    splits = data / "processed" / "splits"
    splits.mkdir(parents=True)
    internal = pd.DataFrame(
        {
            "path": ["raw/internal/real/a.mp4", "raw/internal/fake/b.mp4"],
            "label": [0, 1],
            "split": "test",
            "group": ["a", "b"],
        }
    )
    internal.to_csv(splits / "internal.csv", index=False)
    isolated_models_dir(tmp_path)
    cmd = [sys.executable, "scripts/evaluate_models.py", "--allow-untrained"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                            env=_env(tmp_path, DEEPSECURE_VISUAL_ARCH="test_resnet",
                                     DEEPSECURE_DEVICE="cpu"))  # fmt: skip
    assert result.returncode == 0, result.stderr
    table = pd.read_csv(tmp_path / "reports" / "modality_metrics.csv").set_index(
        ["modality", "dataset"]
    )
    assert table.loc[("lipsync", "internal"), "n"] == 2
    assert table.loc[("lipsync", "internal"), "roc_auc"] == 1.0  # lag clip scores higher
    assert not table.loc[("visual", "internal"), "trained"]
    assert (tmp_path / "reports" / "scores_internal_all.csv").is_file()


# --- held-out talking-face clips (RAVDESS actors 07-08) -------------------------------------


def test_heldout_av_sets_false_alarms_and_constructed_desyncs() -> None:
    from deepsecure.exceptions import InvalidInputError
    from deepsecure.models.base import unavailable
    from deepsecure.models.evaluation import heldout_av_sets
    from deepsecure.preprocessing.audio_extractor import AudioExtractor
    from deepsecure.schemas import ModalityScore
    from deepsecure.utils.synthetic_media import synthetic_voice

    names = ["a.mp4", "bad.mp4", "b.mp4", "c.mp4"]
    ax = AudioExtractor(Settings(_env_file=None))
    audio_of = {n: ax.features_from_waveform(synthetic_voice(False, i, 1.0), 16_000)
                for i, n in enumerate(["a.mp4", "b.mp4", "c.mp4"])}  # fmt: skip
    owner = {id(a): n for n, a in audio_of.items()}

    class Frames:
        def extract(self, p: Path):
            if p.name == "bad.mp4":
                raise InvalidInputError("corrupt")
            return None

    class Audio:
        def extract(self, p: Path):
            return audio_of[p.name]

    class Det:
        is_trained = True

        def __init__(self, name: str) -> None:
            self.name, self.calls = name, []

        def predict(self, x) -> ModalityScore:
            if self.name == "audio" and owner[id(x)] == "b.mp4":
                return unavailable("audio", "too short")
            if self.name == "lipsync":
                self.calls.append((x.audio_shift_s, owner[id(x.audio)]))
            return ModalityScore(name=self.name, fake_prob=0.2, confidence=1.0)

    lip = Det("lipsync")
    thr = {m: (0.5, "dev") for m in ("visual", "audio", "lipsync")}
    keys = ["s1", "s2", "s1", "s2"]  # "bad" is skipped: keys must stay aligned
    sets = {e.modality: e for e in heldout_av_sets(
        [Path(n) for n in names], Frames(), Audio(), Det("visual"), Det("audio"), lip, thr,
        np.random.default_rng(0), "ravdess_heldout", swap_keys=keys)}  # fmt: skip
    assert sets["visual"].ids == ["a.mp4", "b.mp4", "c.mp4"]  # the corrupt clip is skipped
    assert list(sets["visual"].labels) == [0, 0, 0] and sets["audio"].ids == ["a.mp4", "c.mp4"]
    # each clip: itself (real) + shifted (fake) + another sentence's audio (fake)
    assert list(sets["lipsync"].labels) == [0, 1, 1] * 3
    shifts = [abs(s) for s, _ in lip.calls if s != 0]
    assert len(shifts) == 3 and all(0.3 <= s <= 1.5 for s in shifts)
    swaps = [o for s, o in lip.calls[2::3]]
    assert swaps == ["c.mp4", "c.mp4", swaps[2]] and swaps[2] in {"a.mp4", "b.mp4"}
    assert all(e.dataset == "ravdess_heldout" and e.threshold == 0.5 for e in sets.values())

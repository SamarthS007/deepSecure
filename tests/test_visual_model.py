"""Visual detector: helpers, demo mode, checkpoints, training loop and data preparation.

Uses timm's tiny ``test_resnet`` (0.37 M parameters, random init) on the CPU, so the
tests need no internet, no GPU and no dataset.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from deepsecure.config.settings import Settings
from deepsecure.models.base import BaseDetector
from deepsecure.models.visual_model import (
    VisualDetector,
    aggregate_probs,
    build_model,
    crops_to_tensor,
    save_checkpoint,
    visual_confidence,
)
from deepsecure.models.visual_train import build_transforms, train_visual, video_level
from deepsecure.schemas import FrameBatch
from deepsecure.utils.synthetic_media import synthetic_face_crops
from scripts.prepare_faces import main as prepare_faces_main
from scripts.prepare_faces import video_key
from tests.conftest import needs_mediapipe_assets

TINY = "test_resnet"
CPU = torch.device("cpu")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, visual_arch=TINY, device="cpu", models_dir=tmp_path / "models")


def batch_of(crops: np.ndarray) -> FrameBatch:
    n = len(crops)
    return FrameBatch(
        face_crops=crops, crop_timestamps_s=[i * 0.5 for i in range(n)], frames_sampled=max(n, 1),
        faces_found=n, face_found_ratio=1.0 if n else 0.0, timestamps_s=[],
        mouth_open_seq=np.zeros(0, np.float32), mouth_landmarks=np.zeros((0, 40, 2), np.float32),
        fps=25.0,
    )  # fmt: skip


# --- pure helpers ---------------------------------------------------------------------


def test_aggregate_top_k() -> None:
    probs = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    assert aggregate_probs(probs, 2) == pytest.approx(0.85)
    assert aggregate_probs(probs, 10) == pytest.approx(probs.mean())  # k > N -> all
    with pytest.raises(ValueError):
        aggregate_probs(np.array([]), 5)


def test_confidence_single_crop_uses_decisiveness_only() -> None:
    assert visual_confidence(np.array([0.9]), 0.9) == pytest.approx(0.8)
    assert visual_confidence(np.array([0.5]), 0.5) == 0.0  # never "certain" from one 0.5 crop


def test_confidence_drops_when_crops_disagree() -> None:
    agree = visual_confidence(np.array([0.9, 0.9, 0.9]), 0.9)
    disagree = visual_confidence(np.array([0.1, 0.9, 0.9]), 0.9)
    assert disagree < agree


def test_crops_to_tensor_normalises_and_resizes() -> None:
    crops = np.full((2, 299, 299, 3), 255, np.uint8)
    x = crops_to_tensor(crops, 160, [0.5] * 3, [0.5] * 3)
    assert x.shape == (2, 3, 160, 160)
    assert torch.allclose(x, torch.ones_like(x))  # (1 - 0.5) / 0.5


def test_build_model_falls_back() -> None:
    model, arch = build_model("no_such_model", pretrained=False, fallback=TINY)
    assert arch == TINY and model.get_classifier().out_features == 1


def test_train_transform_is_random_and_val_is_not() -> None:
    from PIL import Image

    img = Image.fromarray(synthetic_face_crops(1, fake=False)[0])
    train_t = build_transforms(160, [0.5] * 3, [0.5] * 3, train=True)
    val_t = build_transforms(160, [0.5] * 3, [0.5] * 3, train=False)
    assert torch.equal(val_t(img), val_t(img))
    torch.manual_seed(0)
    outs = [train_t(img) for _ in range(4)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])
    assert outs[0].shape == (3, 160, 160)


# --- detector -------------------------------------------------------------------------


def test_demo_mode(settings: Settings) -> None:
    det = VisualDetector(settings, device=CPU)
    assert isinstance(det, BaseDetector)
    det.load()  # no checkpoint -> random weights
    score = det.predict(batch_of(synthetic_face_crops(6, fake=False)))
    assert score.available and score.trained is False and not det.is_trained
    assert 0.0 <= score.fake_prob <= 1.0
    assert score.details["n_crops"] == 6 and score.details["top_k"] == 5


def test_no_face_is_unavailable(settings: Settings) -> None:
    det = VisualDetector(settings, device=CPU)
    score = det.predict(batch_of(np.zeros((0, 299, 299, 3), np.uint8)))
    assert not score.available and "no face" in score.reason
    assert not det.predict(None).available


def test_checkpoint_round_trip(settings: Settings, tmp_path: Path) -> None:
    torch.manual_seed(1)
    model, arch = build_model(TINY, pretrained=False)
    crops = synthetic_face_crops(3, fake=True)

    real = save_checkpoint(model, arch, tmp_path / "real.pt", data_source="data/x/index.csv",
                           val_video_auc=0.9)  # fmt: skip
    det = VisualDetector(settings, device=CPU)
    det.load(real)
    assert det.is_trained and det.checkpoint_info["val_video_auc"] == 0.9
    with torch.inference_mode():
        expected = torch.sigmoid(model.eval()(crops_to_tensor(crops, **det.cfg))).squeeze(1)
    assert np.allclose(det.crop_probs(crops), expected.numpy(), atol=1e-5)
    assert det.predict(batch_of(crops)).trained is True

    synthetic = save_checkpoint(model, arch, tmp_path / "syn.pt", data_source="synthetic")
    det.load(synthetic)
    assert not det.is_trained  # synthetic checkpoints never end demo mode


def test_untrusted_pickle_is_refused(settings: Settings, tmp_path: Path) -> None:
    """weights_only=True: a checkpoint containing arbitrary objects must not load."""
    bad = tmp_path / "bad.pt"
    torch.save({"state_dict": {}, "arch": TINY, "payload": object()}, bad)
    with pytest.raises(Exception, match="(?i)weights_only|unsupported|global"):
        VisualDetector(settings, device=CPU).load(bad)


# --- training -------------------------------------------------------------------------


def _toy_split(root: Path, n_videos: int, split: str, offset: int) -> pd.DataFrame:
    from PIL import Image

    rows = []
    for label in (0, 1):
        for v in range(n_videos):
            key = f"{split}{label}_{v}"
            for i, crop in enumerate(
                synthetic_face_crops(3, bool(label), seed=offset + 100 * label + v)
            ):
                rel = f"{key}_{i}.png"
                Image.fromarray(crop).save(root / rel)
                rows.append({"path": rel, "label": label, "video": key, "group": key})
    return pd.DataFrame(rows)


def test_training_learns_the_toy_artifact(tmp_path: Path) -> None:
    train_df = _toy_split(tmp_path, 6, "train", 0)
    val_df = _toy_split(tmp_path, 3, "val", 5000)
    torch.manual_seed(0)
    model, arch = build_model(TINY, pretrained=False)
    out = tmp_path / "best.pt"
    result = train_visual(
        model, arch, train_df, val_df, CPU, out, epochs=4, lr=3e-3, weight_decay=0.0,
        freeze_epochs=1, patience=4, batch_size=8, top_k=5, root=tmp_path,
        metadata={"data_source": "synthetic"},
    )  # fmt: skip
    hist = result.history
    assert hist[0]["frozen_backbone"] and not hist[-1]["frozen_backbone"]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert result.best_val_auc >= 0.8
    assert out.is_file() and result.checkpoint == out
    bundle = torch.load(out, weights_only=True)
    assert bundle["best_epoch"] == result.best_epoch and bundle["data_source"] == "synthetic"


def test_training_needs_both_classes(tmp_path: Path) -> None:
    df = _toy_split(tmp_path, 2, "train", 0)
    real_only = df[df["label"] == 0]
    model, arch = build_model(TINY, pretrained=False)
    with pytest.raises(ValueError, match="both"):
        train_visual(model, arch, real_only, real_only, CPU, tmp_path / "x.pt", epochs=1,
                     lr=1e-3, weight_decay=0, freeze_epochs=0, patience=1, batch_size=4,
                     top_k=5, root=tmp_path)  # fmt: skip


def test_video_level_matches_inference_aggregation() -> None:
    df = pd.DataFrame({"video": ["a", "a", "b", "b"], "label": [1, 1, 0, 0]})
    labels, scores = video_level(df, np.array([0.2, 0.8, 0.1, 0.3]), top_k=1)
    assert list(labels) == [1, 0] and np.allclose(scores, [0.8, 0.3])


# --- data preparation -----------------------------------------------------------------


def test_video_key_is_unique_across_methods() -> None:
    a = video_key("raw/ff/manipulated_sequences/Deepfakes/c23/videos/000_003.mp4")
    b = video_key("raw/ff/manipulated_sequences/FaceSwap/c23/videos/000_003.mp4")
    assert a != b and a.startswith("000_003_")


def test_prepare_synthetic(tmp_path: Path) -> None:
    assert prepare_faces_main(["--synthetic", "4", "--out", str(tmp_path)]) == 0
    index = pd.read_csv(tmp_path / "index.csv")
    assert len(index) == 2 * 4 * 4  # 2 classes x 4 videos x 4 crops
    assert set(index["split"]) == {"train", "val"}
    assert (tmp_path / index["path"].iloc[0]).is_file()


@needs_mediapipe_assets
def test_prepare_from_split_csv(tmp_path: Path, face_video: Path, silent_video: Path) -> None:
    """A split CSV with a face video and a no-face video -> crops only for the face video."""
    csv = tmp_path / "splits.csv"
    pd.DataFrame([
        {"path": face_video.name, "label": 0, "split": "train", "group": "g1", "dataset": "t"},
        {"path": silent_video.name, "label": 1, "split": "val", "group": "g2", "dataset": "t"},
    ]).to_csv(csv, index=False)  # fmt: skip
    out = tmp_path / "faces"
    env = {
        **os.environ,
        "DEEPSECURE_DATA_DIR": str(face_video.parent),
    }  # paths are data_dir-relative
    cmd = [sys.executable, "scripts/prepare_faces.py", "--splits-csv", str(csv), "--out", str(out),
           "--crops-per-video", "2"]  # fmt: skip
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=Path(__file__).parents[1]
    )
    assert result.returncode == 0, result.stderr
    assert "no face found" in result.stdout
    index = pd.read_csv(out / "index.csv")
    assert len(index) == 2 and set(index["split"]) == {"train"}  # no-face video skipped

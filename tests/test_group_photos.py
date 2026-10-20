"""Group photos: tiled face scan, small-face correction and the visual detector's group path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from deepsecure.config.settings import Settings
from deepsecure.models.small_faces import (
    SMALL_FACE_FILE,
    fit_table,
    group_fake_prob,
    logit,
    shrink,
    small_face_z,
)
from deepsecure.models.visual_model import VisualDetector, build_model, save_checkpoint
from deepsecure.preprocessing.frame_extractor import FrameExtractor, merge_boxes, tile_starts
from deepsecure.schemas import FrameBatch
from deepsecure.utils.synthetic_media import (
    synthetic_face_crops,
    write_face_photo,
    write_group_photo,
    write_no_face_image,
)
from tests.conftest import needs_mediapipe_assets

# genuine faces: logit mean -4 / std 1 at full size (300 px), -1 / 1 at 64 px
TABLE = {
    "source_px": [64.0, 300.0],
    "mu": [-1.0, -4.0],
    "sigma": [1.0, 1.0],
    "reference": {"mu": -4.0, "sigma": 1.0},
}


def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


# --- tiling ------------------------------------------------------------------------------


def test_tile_starts_cover_the_image() -> None:
    assert tile_starts(500, 640, 320) == [0]
    assert tile_starts(1600, 640, 320) == [0, 320, 640, 960]
    assert tile_starts(1700, 640, 320) == [0, 320, 640, 960, 1060]  # last tile flush


def test_merge_boxes_drops_the_same_face_seen_in_two_tiles() -> None:
    whole = (0.9, 100.0, 100.0, 80.0, 80.0)
    cut_off = (0.7, 100.0, 100.0, 50.0, 80.0)  # the same face cut at a tile edge
    other = (0.8, 400.0, 100.0, 80.0, 80.0)
    assert merge_boxes([cut_off, other, whole]) == [whole, other]


# --- small-face maths ---------------------------------------------------------------------


def test_shrink_simulates_a_smaller_face_like_the_app_crops_it() -> None:
    region = synthetic_face_crops(1, fake=False, size=200)[0]  # a 200 px face region
    small, same = shrink(region, 64), shrink(region, 200)
    assert small.shape == same.shape == (299, 299, 3) and small.dtype == np.uint8
    detail = lambda x: np.abs(np.diff(x.astype(float), axis=1)).mean()  # noqa: E731
    assert detail(small) < detail(same)
    assert np.array_equal(shrink(region, 400), same)  # can't add detail: the normal crop


def test_fit_table_uses_sizes_below_the_reference() -> None:
    rng = np.random.default_rng(0)
    probs = {64: rng.uniform(0.1, 0.6, 50), 400: rng.uniform(0.0, 0.2, 50)}
    ref = rng.uniform(0.0, 0.1, 50)
    t = fit_table(probs, ref, 300.0, checkpoint_sha256="x")
    assert t["source_px"] == [64.0, 300.0] and t["checkpoint_sha256"] == "x"
    assert t["mu"][0] == pytest.approx(logit(probs[64]).mean())
    assert t["reference"]["mu"] == pytest.approx(logit(ref).mean()) == t["mu"][-1]
    with pytest.raises(ValueError):
        fit_table({400: ref}, ref, 300.0)


def test_typical_genuine_face_maps_to_a_typical_portrait_score() -> None:
    # a single face exactly as unusual as an average genuine face of its size
    for px, mu in ((64.0, -1.0), (300.0, -4.0)):
        p, worst = group_fake_prob(np.array([sigmoid(mu)]), np.array([px]), TABLE)
        assert worst == 0 and p == pytest.approx(sigmoid(-4.0), abs=1e-6)


def test_the_same_score_is_less_suspicious_on_a_smaller_face() -> None:
    p_small, _ = group_fake_prob(np.array([0.5]), np.array([64.0]), TABLE)
    p_large, _ = group_fake_prob(np.array([0.5]), np.array([300.0]), TABLE)
    assert p_small < 0.40 < p_large
    z = small_face_z(np.array([0.5, 0.5]), np.array([64.0, 128.0]), TABLE)
    assert z[0] < z[1]  # 128 px is interpolated between the two grid points


def test_more_faces_raise_the_bar() -> None:
    worst = 0.6
    one, _ = group_fake_prob(np.array([worst]), np.array([64.0]), TABLE)
    many, idx = group_fake_prob(np.array([0.1] * 9 + [worst]), np.array([64.0] * 10), TABLE)
    assert idx == 9 and many < one


def test_a_clearly_fake_face_stands_out_without_overflow() -> None:
    p, idx = group_fake_prob(np.array([0.2, 1 - 1e-6, 0.3]), np.array([64.0, 64.0, 64.0]), TABLE)
    assert idx == 1 and 0.9 < p <= 1.0 and np.isfinite(p)


# --- visual detector, group path -----------------------------------------------------------


def group_batch(n: int = 3) -> FrameBatch:
    return FrameBatch(
        face_crops=synthetic_face_crops(n, fake=False), crop_timestamps_s=[0.0] * n,
        frames_sampled=1, faces_found=n, face_found_ratio=1.0, timestamps_s=[],
        mouth_open_seq=np.zeros(0, np.float32), mouth_landmarks=np.zeros((0, 40, 2), np.float32),
        fps=0.0, group_scan=True, crop_source_px=[80.0, 120.0, 160.0][:n],
    )  # fmt: skip


@pytest.fixture
def trained(tmp_path: Path) -> tuple[Settings, Path]:
    s = Settings(_env_file=None, visual_arch="test_resnet", device="cpu",
                 models_dir=tmp_path / "models")  # fmt: skip
    torch.manual_seed(0)
    model, arch = build_model("test_resnet", pretrained=False)
    ckpt = save_checkpoint(model, arch, s.models_dir / "visual" / "visual_best.pt",
                           data_source="data/x/index.csv")  # fmt: skip
    return s, ckpt


def test_group_photo_needs_the_calibration_of_this_model(trained) -> None:
    s, ckpt = trained
    det = VisualDetector(s, device=torch.device("cpu"))
    det.load()
    score = det.predict(group_batch())
    assert not score.available and "calibrate_small_faces.py" in score.reason

    table = s.models_dir / "visual" / SMALL_FACE_FILE
    table.write_text(json.dumps({**TABLE, "checkpoint_sha256": "another model"}))
    det.load()
    assert not det.predict(group_batch()).available

    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    table.write_text(json.dumps({**TABLE, "checkpoint_sha256": digest}))
    det.load()
    score = det.predict(group_batch())
    assert score.available and score.trained
    assert score.details["group_photo"] == 1.0 and score.details["n_crops"] == 3
    probs = det.crop_probs(group_batch().face_crops)
    expected, worst = group_fake_prob(probs, np.array([80.0, 120.0, 160.0]), TABLE)
    assert score.fake_prob == pytest.approx(expected) and score.details["worst_face"] == worst


def test_untrained_model_never_scores_group_photos(tmp_path: Path) -> None:
    s = Settings(_env_file=None, visual_arch="test_resnet", device="cpu",
                 models_dir=tmp_path / "models")  # fmt: skip
    det = VisualDetector(s, device=torch.device("cpu"))
    det.load()  # demo mode: no checkpoint, so no calibration can match
    assert not det.predict(group_batch()).available


# --- tiled scan on images ---------------------------------------------------------------------


@needs_mediapipe_assets
def test_group_photo_is_scanned_face_by_face(tmp_path: Path) -> None:
    with FrameExtractor() as fx:
        group = fx.extract_image(write_group_photo(tmp_path / "group.jpg"))
        portrait = fx.extract_image(write_face_photo(tmp_path / "face.jpg"))
        empty = fx.extract_image(write_no_face_image(tmp_path / "none.png", (1400, 900)))
    assert group.group_scan and group.faces_found == 4 and group.face_found_ratio == 1.0
    assert group.face_crops.shape == (4, 299, 299, 3) and len(group.crop_source_px) == 4
    assert all(80 < px < 200 for px in group.crop_source_px)
    assert not portrait.group_scan and portrait.faces_found == 1
    assert not empty.group_scan and empty.faces_found == 0

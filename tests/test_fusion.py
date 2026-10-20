"""Fusion engine (weighted + learned head) and the fusion training script."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deepsecure.config.settings import Settings
from deepsecure.fusion.fusion_engine import NO_EVIDENCE, FusionEngine
from deepsecure.fusion.learned_head import (
    FEATURE_NAMES,
    LearnedFusionHead,
    features_from_probs,
)
from deepsecure.schemas import ModalityScore
from deepsecure.utils.synthetic_media import synthetic_fusion_scores

ROOT = Path(__file__).resolve().parents[1]


def sc(name: str, p: float, conf: float = 1.0, available: bool = True, trained: bool = True):
    return ModalityScore(name=name, fake_prob=p, confidence=conf, available=available,
                         trained=trained)  # fmt: skip


@pytest.fixture
def engine() -> FusionEngine:
    return FusionEngine(Settings(_env_file=None))


# --- weighted fusion ---------------------------------------------------------------------


def test_equal_confidence_uses_base_weights(engine: FusionEngine) -> None:
    r = engine.fuse([sc("visual", 0.8), sc("audio", 0.4), sc("lipsync", 0.6)], "video")
    assert r.fused_fake_prob == pytest.approx(0.45 * 0.8 + 0.35 * 0.4 + 0.20 * 0.6)
    assert r.weights_used == pytest.approx({"visual": 0.45, "audio": 0.35, "lipsync": 0.20})
    assert r.method == "weighted" and r.coverage == 1.0 and r.trained


def test_weights_renormalise_over_available(engine: FusionEngine) -> None:
    r = engine.fuse([sc("visual", 0.9), sc("audio", 0.5, 0, available=False),
                     sc("lipsync", 0.5, 0, available=False)], "video")  # fmt: skip
    assert r.weights_used == {"visual": 1.0} and r.fused_fake_prob == pytest.approx(0.9)
    assert r.available_modalities == ["visual"]


def test_zero_confidence_score_gets_no_weight(engine: FusionEngine) -> None:
    r = engine.fuse([sc("visual", 0.9, 0.8), sc("audio", 0.5, 0.0, trained=False)], "video")
    assert r.weights_used["audio"] == 0.0 and r.fused_fake_prob == pytest.approx(0.9)
    assert any("audio: untrained" in w for w in r.warnings)
    assert not r.trained


def test_undecided_low_confidence_score_does_not_drag_confidence(engine: FusionEngine) -> None:
    # measured pattern on genuine RAVDESS videos: confident real face and voice, lip-sync
    # undecided (≈ 0.47, confidence ≈ 0.15); evidence confidence is weighted like the fusion
    r = engine.fuse([sc("visual", 0.05, 0.9), sc("audio", 0.01, 0.7), sc("lipsync", 0.47, 0.15)],
                    "video")  # fmt: skip
    w = np.array([0.45 * 0.9, 0.35 * 0.7, 0.20 * 0.15])
    w /= w.sum()
    p = np.array([0.05, 0.01, 0.47])
    spread = np.sqrt(np.dot(w, (p - np.dot(w, p)) ** 2))
    expected = (1 - spread / 0.5) * np.dot(w, [0.9, 0.7, 0.15])
    assert r.confidence == pytest.approx(expected) and r.confidence > 0.5


def test_all_zero_confidence_falls_back_to_base_weights(engine: FusionEngine) -> None:
    r = engine.fuse([sc("visual", 0.6, 0.0), sc("audio", 0.4, 0.0)], "video")
    assert r.fused_fake_prob == pytest.approx((0.45 * 0.6 + 0.35 * 0.4) / 0.8)
    assert r.confidence == 0.0


def test_confidence_weighting_can_be_switched_off() -> None:
    plain = FusionEngine(Settings(_env_file=None, fusion_confidence_weighting=False))
    r = plain.fuse([sc("visual", 1.0, 0.1), sc("audio", 0.0, 0.9)], "video")
    assert r.fused_fake_prob == pytest.approx(0.45 / 0.8)


def test_no_evidence(engine: FusionEngine) -> None:
    r = engine.fuse([sc("visual", 0.5, 0, available=False)], "video")
    assert r.fused_fake_prob == 0.5 and r.confidence == 0.0
    assert r.warnings == [NO_EVIDENCE] and r.max_modality is None


# --- confidence: coverage, agreement ---------------------------------------------------------


def test_photo_is_not_penalised_for_having_no_voice(engine: FusionEngine) -> None:
    photo = engine.fuse([sc("visual", 0.05, 0.9)], "image")
    silent_video = engine.fuse([sc("visual", 0.05, 0.9)], "video")
    assert photo.coverage == 1.0 and photo.confidence == pytest.approx(0.9)
    assert silent_video.coverage == pytest.approx(0.45)
    assert silent_video.confidence == pytest.approx(0.45 * 0.9)


def test_disagreement_lowers_confidence_and_warns(engine: FusionEngine) -> None:
    agree = engine.fuse([sc("visual", 0.9, 0.8), sc("audio", 0.85, 0.8)], "video")
    clash = engine.fuse([sc("visual", 0.9, 0.8), sc("audio", 0.1, 0.8)], "video")
    assert clash.agreement < 0.3 < agree.agreement
    assert clash.confidence < agree.confidence
    assert any("disagree: visual says fake (0.90) but audio says real (0.10)" in w
               for w in clash.warnings)  # fmt: skip
    assert clash.max_modality == "visual" and clash.max_fake_prob == 0.9


def test_unsure_scores_do_not_trigger_disagreement(engine: FusionEngine) -> None:
    r = engine.fuse([sc("visual", 0.9, 0.8), sc("audio", 0.1, 0.1)], "video")
    assert not any("disagree" in w for w in r.warnings)


# --- learned head -------------------------------------------------------------------------------


def test_features_encode_missing_modalities() -> None:
    f = features_from_probs({"visual": 0.5, "audio": None, "lipsync": float("nan")})
    assert len(f) == len(FEATURE_NAMES) == 6
    assert list(f) == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # logit(0.5) = 0, available flag 1


def test_head_learns_to_ignore_noise() -> None:
    head = LearnedFusionHead().fit(synthetic_fusion_scores(600, seed=1))
    w = head.modality_weights()
    assert w["lipsync"] < 0.1 < w["audio"] < w["visual"]
    assert sum(w.values()) == pytest.approx(1.0)


def test_head_needs_both_classes() -> None:
    df = synthetic_fusion_scores(50)
    with pytest.raises(ValueError, match="both"):
        LearnedFusionHead().fit(df[df["label"] == 1])


def test_engine_uses_learned_head_and_falls_back(tmp_path: Path) -> None:
    s = Settings(_env_file=None, fusion_method="learned")
    scores = [sc("visual", 0.9), sc("audio", 0.8), sc("lipsync", 0.5, 0, available=False)]

    fallback = FusionEngine(s).fuse(scores, "video")
    assert fallback.method == "weighted"
    assert any("no fitted head" in w for w in fallback.warnings)

    head = LearnedFusionHead().fit(synthetic_fusion_scores(400))
    path = head.save(tmp_path / "head.joblib", data_source="synthetic")
    loaded = LearnedFusionHead()
    assert loaded.load(path) and not loaded.is_trained  # synthetic never ends demo mode
    r = FusionEngine(s, head=loaded).fuse(scores, "video")
    assert r.method == "learned" and r.fused_fake_prob > 0.5
    assert set(r.weights_used) == {"visual", "audio", "lipsync"}


def test_head_load_missing_and_mismatched(tmp_path: Path) -> None:
    import joblib

    assert LearnedFusionHead().load(tmp_path / "none.joblib") is False
    joblib.dump({"model": None, "feature_names": ("x",)}, tmp_path / "old.joblib")
    with pytest.raises(ValueError, match="different features"):
        LearnedFusionHead().load(tmp_path / "old.joblib")


# --- the script --------------------------------------------------------------------------------


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "DEEPSECURE_MODELS_DIR": str(tmp_path / "models"),
           "DEEPSECURE_REPORTS_DIR": str(tmp_path / "reports")}  # fmt: skip
    return subprocess.run([sys.executable, "scripts/train_fusion.py", *args],
                          capture_output=True, text=True, env=env, cwd=ROOT)  # fmt: skip


def test_script_on_a_scores_table(tmp_path: Path) -> None:
    csv = tmp_path / "scores.csv"
    synthetic_fusion_scores(200, seed=3).to_csv(csv, index=False)
    result = _run(tmp_path, "--scores", str(csv), "--save")
    assert result.returncode == 0, result.stderr
    table = pd.read_csv(tmp_path / "reports" / "fusion_comparison.csv")
    assert list(table["method"])[:2] == ["weighted", "learned (CV)"]
    assert {"visual only", "audio only", "lipsync only"} <= set(table["method"])
    assert (table["roc_auc"].iloc[:2] > 0.8).all()
    assert (tmp_path / "models" / "fusion" / "fusion_head.joblib").is_file()


def test_script_refuses_too_little_data(tmp_path: Path) -> None:
    csv = tmp_path / "tiny.csv"
    df = synthetic_fusion_scores(40)
    df.loc[df["label"] == 1, "group"] = "same-person"  # one identity group for every fake
    df.to_csv(csv, index=False)
    result = _run(tmp_path, "--scores", str(csv))
    assert result.returncode == 1 and "identity groups per class" in result.stdout


def test_cv_never_puts_a_group_on_both_sides() -> None:
    from sklearn.model_selection import StratifiedGroupKFold

    df = synthetic_fusion_scores(100)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(
        df, df["label"], df["group"]
    ):
        assert not set(df["group"].iloc[tr]) & set(df["group"].iloc[te])
    assert np.isnan(df.loc[df["media_type"] == "image", "audio"]).all()

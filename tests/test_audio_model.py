"""Audio detector: EER, model selection, demo mode, checkpoints, silence shortcut, script."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from deepsecure.config.settings import Settings
from deepsecure.models.audio_model import AudioDetector, save_classifier, select_model
from deepsecure.models.base import BaseDetector
from deepsecure.models.metrics import compute_eer
from deepsecure.preprocessing.audio_extractor import AudioExtractor
from deepsecure.utils.synthetic_media import synthetic_voice

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, models_dir=tmp_path / "models")


def toy_features(n: int, settings: Settings, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    ax = AudioExtractor(settings)
    x, y = [], []
    for label in (0, 1):
        for i in range(n):
            wave = synthetic_voice(bool(label), seed=offset + 1000 * label + i, duration_s=2.0)
            x.append(ax.features_from_waveform(wave, 16_000).summary_vector)
            y.append(label)
    return np.stack(x).astype(np.float64), np.array(y)


# --- metrics ----------------------------------------------------------------------------


def test_eer_perfect_and_reversed() -> None:
    y = np.array([0, 0, 1, 1])
    assert compute_eer(y, np.array([0.1, 0.2, 0.8, 0.9]))[0] == 0.0
    assert compute_eer(y, np.array([0.9, 0.8, 0.2, 0.1]))[0] == 1.0


def test_eer_is_about_half_for_random_scores() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    eer, _ = compute_eer(y, rng.random(4000))
    assert eer == pytest.approx(0.5, abs=0.03)


def test_eer_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        compute_eer(np.zeros(3), np.array([0.1, 0.2, 0.3]))


# --- the silence shortcut ---------------------------------------------------------------


def test_features_ignore_leading_silence(settings: Settings) -> None:
    """ASVspoof 2019 bona fide clips have longer leading silence; features must not see it."""
    ax = AudioExtractor(settings)
    voice = synthetic_voice(spoof=False, seed=3, duration_s=2.0)
    short = ax.features_from_waveform(np.concatenate([np.zeros(1600), voice]), 16_000)
    long = ax.features_from_waveform(np.concatenate([np.zeros(16_000), voice]), 16_000)
    assert long.offset_s - short.offset_s == pytest.approx(0.9, abs=0.02)
    assert long.silence_ratio > short.silence_ratio  # the raw signal *does* differ...
    np.testing.assert_allclose(long.summary_vector, short.summary_vector, rtol=1e-3, atol=1e-3)


# --- model selection and detector ------------------------------------------------------


def test_select_model_compares_both_candidates(settings: Settings) -> None:
    x_tr, y_tr = toy_features(15, settings)
    x_va, y_va = toy_features(6, settings, offset=500)
    best, model, results = select_model(x_tr, y_tr, x_va, y_va, settings)
    assert set(results) == {"logreg", "mlp"} and best in results
    assert results[best]["eer"] == min(r["eer"] for r in results.values())
    assert results[best]["eer"] <= 0.1  # the toy spoof traces are easy
    assert model.predict_proba(x_va).shape == (len(y_va), 2)


def test_demo_mode_is_neutral(settings: Settings) -> None:
    det = AudioDetector(settings)
    assert isinstance(det, BaseDetector)
    det.load()
    score = det.predict(AudioExtractor(settings).features_from_waveform(
        synthetic_voice(False, 1), 16_000))  # fmt: skip
    assert score.available and not score.trained
    assert score.fake_prob == 0.5 and score.confidence == 0.0 and score.details["demo"] == 1.0


def test_missing_audio_is_unavailable(settings: Settings) -> None:
    score = AudioDetector(settings).predict(None)
    assert not score.available and score.reason == "no usable audio"


def test_round_trip_and_scores(settings: Settings, tmp_path: Path) -> None:
    x_tr, y_tr = toy_features(15, settings)
    _, model, results = select_model(x_tr, y_tr, *toy_features(6, settings, 500), settings)
    path = save_classifier(model, tmp_path / "clf.joblib", settings, data_source="asv.csv",
                           model_name="x", val_eer_threshold=0.5)  # fmt: skip
    det = AudioDetector(settings)
    det.load(path)
    assert det.is_trained
    ax = AudioExtractor(settings)
    real = det.predict(ax.features_from_waveform(synthetic_voice(False, 777), 16_000))
    fake = det.predict(ax.features_from_waveform(synthetic_voice(True, 778), 16_000))
    assert real.trained and real.fake_prob < 0.5 < fake.fake_prob
    assert fake.details["above_eer_threshold"] == 1.0

    save_classifier(model, tmp_path / "syn.joblib", settings, data_source="synthetic")
    det.load(tmp_path / "syn.joblib")
    assert not det.is_trained


def test_short_audio_lowers_confidence(settings: Settings, tmp_path: Path) -> None:
    x_tr, y_tr = toy_features(15, settings)
    _, model, _ = select_model(x_tr, y_tr, *toy_features(6, settings, 500), settings)
    det = AudioDetector(settings)
    det.load(save_classifier(model, tmp_path / "c.joblib", settings, data_source="x"))
    ax = AudioExtractor(settings)
    wave = synthetic_voice(True, 9, duration_s=4.0)
    long = det.predict(ax.features_from_waveform(wave, 16_000))
    short = det.predict(ax.features_from_waveform(wave[:16_000], 16_000))
    assert short.confidence < long.confidence


def test_mfcc_mismatch_is_refused(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "old.joblib"
    joblib.dump({"model": None, "n_mfcc": 13}, path)
    with pytest.raises(ValueError, match="n_mfcc"):
        AudioDetector(settings).load(path)


# --- the training script on a tiny fake ASVspoof layout ---------------------------------


def test_train_script_real_data_path(tmp_path: Path) -> None:
    data = tmp_path / "data"
    rows = []
    for split, n, off in (("train", 8, 0), ("val", 4, 900)):
        for label in (0, 1):
            for i in range(n):
                rel = f"raw/asv/{split}_{label}_{i}.flac"
                (data / rel).parent.mkdir(parents=True, exist_ok=True)
                sf.write(
                    data / rel, synthetic_voice(bool(label), off + 100 * label + i, 2.0), 16_000
                )
                rows.append({"path": rel, "label": label, "split": split, "group": f"spk{i}",
                             "exists": True})  # fmt: skip
    csv = data / "processed" / "splits" / "asvspoof2019_la.csv"
    csv.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(csv, index=False)

    env = {**os.environ, "DEEPSECURE_DATA_DIR": str(data),
           "DEEPSECURE_MODELS_DIR": str(tmp_path / "models"),
           "DEEPSECURE_REPORTS_DIR": str(tmp_path / "reports")}  # fmt: skip
    cmd = [sys.executable, "scripts/train_audio.py", "--n-jobs", "2"]
    first = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "extracted 16/16" in first.stdout and "kept" in first.stdout
    assert (tmp_path / "models" / "audio" / "audio_classifier.joblib").is_file()
    assert (tmp_path / "reports" / "audio_model_selection.csv").is_file()

    second = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    assert "cached feature vectors" in second.stdout  # the cache is reused


def test_summary_features_and_cache(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from deepsecure.models.audio_model import summary_features

    s = Settings(_env_file=None, data_dir=tmp_path)
    rows = []
    for i, label in enumerate([0, 1, 0, 1]):
        sf.write(tmp_path / f"{i}.flac", synthetic_voice(bool(label), i, 2.0), 16_000)
        rows.append({"path": f"{i}.flac", "label": label, "split": "test"})
    (tmp_path / "bad.flac").write_bytes(b"not audio")
    rows.append({"path": "bad.flac", "label": 1, "split": "test"})
    df = pd.DataFrame(rows)

    x, y = summary_features(df, "test", s, n_jobs=1, cache_dir=tmp_path / "cache")
    assert x.shape == (4, 170) and list(y) == [0, 1, 0, 1]  # the broken file is skipped
    assert "extracted 4/5" in capsys.readouterr().out
    x2, _ = summary_features(df, "test", s, n_jobs=1, cache_dir=tmp_path / "cache")
    assert "cached" in capsys.readouterr().out and np.array_equal(x, x2)
    with pytest.raises(ValueError, match="no usable audio"):
        summary_features(df[df["path"] == "bad.flac"], "test", s, 1, tmp_path / "cache")

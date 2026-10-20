"""Lip-sync baseline: signal helpers, availability rules, scoring, calibration, real clips."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from deepsecure.config.settings import Settings
from deepsecure.models.base import BaseDetector
from deepsecure.models.lipsync_model import (
    LipSyncDetector,
    calibration_rows,
    energy_on_grid,
    even_grid,
    fit_calibration,
    lag_correlations,
    mouth_on_grid,
    save_calibration,
)
from deepsecure.preprocessing.audio_extractor import AudioExtractor
from deepsecure.preprocessing.frame_extractor import FrameExtractor
from deepsecure.preprocessing.mediapipe_assets import assets_status
from deepsecure.schemas import AudioFeatures, FrameBatch, LipSyncInput
from deepsecure.utils.synthetic_media import speech_rhythm_envelope, write_talking_face_clip
from tests.conftest import needs_mediapipe_assets

# --- helpers: build contracts straight from envelopes (no media files) ----------------


def make_frames(duration_s: float = 4.0, lag_s: float = 0.0, seed: int = 0, fps: float = 15.0,
                static: bool = False) -> FrameBatch:  # fmt: skip
    t = np.arange(0, duration_s, 1 / fps)
    mouth = 0.06 + (0.0 if static else 0.03) * speech_rhythm_envelope(t - lag_s, seed)
    return FrameBatch(
        face_crops=np.zeros((0, 299, 299, 3), np.uint8),
        frames_sampled=0,
        faces_found=0,
        face_found_ratio=0.0,
        timestamps_s=[round(x, 4) for x in t],
        mouth_open_seq=mouth.astype(np.float32),
        mouth_landmarks=np.zeros((len(t), 40, 2), np.float32),
        fps=25.0,
    )


def make_audio(duration_s: float = 4.0, seed: int = 0, offset_s: float = 0.0) -> AudioFeatures:
    """Energy envelope at 100 frames/s that follows the same rhythm, starting at offset_s."""
    k = np.arange(int((duration_s - offset_s) * 100))
    env = speech_rhythm_envelope(offset_s + k / 100, seed) + 0.01
    return AudioFeatures(
        waveform=np.zeros(16_000, np.float32),
        mfcc=np.zeros((40, len(k)), np.float32),
        log_mel=np.zeros((64, len(k)), np.float32),
        energy_envelope=env.astype(np.float32),
        summary_vector=np.zeros(170, np.float32),
        duration_s=duration_s,
        offset_s=offset_s,
    )


@pytest.fixture
def detector() -> LipSyncDetector:
    return LipSyncDetector(Settings(_env_file=None))


# --- signal helpers -------------------------------------------------------------


def test_even_grid() -> None:
    assert np.allclose(even_grid(0.0, 1.0, 4), [0, 0.25, 0.5, 0.75, 1.0])
    assert len(even_grid(1.0, 0.0, 4)) == 0


def test_mouth_on_grid_keeps_face_gaps() -> None:
    t = [0.0, 0.1, 0.2, 0.3, 0.4]
    m = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    out = mouth_on_grid(t, m, np.array([0.0, 0.05, 0.2, 0.4, 0.5]))
    assert out[0] == 1.0 and out[1] == pytest.approx(1.5)
    assert np.isnan(out[2])  # nearest sample had no face: not bridged
    assert out[3] == 5.0 and np.isnan(out[4])  # 0.5 is past the end


def test_energy_on_grid_respects_offset_and_shift() -> None:
    audio = AudioFeatures(
        waveform=np.zeros(10, np.float32), mfcc=np.zeros((40, 3), np.float32),
        log_mel=np.zeros((64, 3), np.float32), summary_vector=np.zeros(170, np.float32),
        energy_envelope=np.ones(100, np.float32), duration_s=3.0, offset_s=1.0,
    )  # fmt: skip
    grid = np.array([0.5, 1.5, 2.5, 3.5])
    e = energy_on_grid(audio, grid)
    assert e[0] == 0.0  # trimmed leading silence
    assert e[1] == pytest.approx(1.0)
    assert e[2] == 0.0  # trimmed trailing silence (energy only covers 1.0-2.0 s)
    assert np.isnan(e[3])  # after the recording ends
    assert np.isnan(energy_on_grid(audio, grid, shift_s=1.0)[0])  # audio starts 1 s later


def test_lag_correlations_find_a_known_lag() -> None:
    rng = np.random.default_rng(0)
    x = np.convolve(rng.standard_normal(300), np.ones(5) / 5, mode="same")
    mouth = np.roll(x, 4)  # mouth moves 4 steps after the sound
    corrs = lag_correlations(mouth, x, max_lag=10, min_overlap=20)
    assert int(np.nanargmax(corrs)) - 10 == 4


# --- availability -----------------------------------------------------------------


def test_detector_follows_the_protocol(detector: LipSyncDetector) -> None:
    assert isinstance(detector, BaseDetector)
    assert detector.name == "lipsync" and detector.is_trained is False


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (LipSyncInput(frames=make_frames(), audio=None), "no usable audio"),
        (LipSyncInput(frames=None, audio=make_audio()), "no mouth-movement timeline"),
        (LipSyncInput(frames=make_frames(static=True), audio=make_audio()), "barely moves"),
        (LipSyncInput(frames=make_frames(duration_s=0.6), audio=make_audio()), "only 0."),
    ],
)
def test_unavailable_cases(detector: LipSyncDetector, data: LipSyncInput, reason: str) -> None:
    score = detector.predict(data)
    assert score.available is False and reason in score.reason
    assert score.confidence == 0.0


def test_image_batch_is_unavailable(detector: LipSyncDetector) -> None:
    image_like = make_frames().model_copy(
        update={"timestamps_s": [], "mouth_open_seq": np.zeros(0, np.float32),
                "mouth_landmarks": np.zeros((0, 40, 2), np.float32)}
    )  # fmt: skip
    assert detector.predict(LipSyncInput(frames=image_like, audio=make_audio())).available is False


def test_no_face_frames_are_unavailable(detector: LipSyncDetector) -> None:
    frames = make_frames()
    frames.mouth_open_seq[:] = np.nan
    score = detector.predict(LipSyncInput(frames=frames, audio=make_audio()))
    assert not score.available and "no face" in score.reason


# --- scoring (uncalibrated rule) -----------------------------------------------------


def test_in_sync_scores_low(detector: LipSyncDetector) -> None:
    score = detector.predict(LipSyncInput(frames=make_frames(), audio=make_audio()))
    assert score.available and score.trained is False
    assert score.fake_prob < 0.1
    assert score.details["best_lag_s"] == pytest.approx(0.0, abs=0.04)


def test_trimmed_audio_stays_aligned(detector: LipSyncDetector) -> None:
    """offset_s must put trimmed audio back in the right place (Phase 3 contract)."""
    score = detector.predict(LipSyncInput(frames=make_frames(), audio=make_audio(offset_s=0.4)))
    assert score.fake_prob < 0.1


@pytest.mark.parametrize("shift", [0.2, 0.3, -0.3, 0.5, 1.0])
def test_shifted_audio_scores_high(detector: LipSyncDetector, shift: float) -> None:
    data = LipSyncInput(frames=make_frames(), audio=make_audio(), audio_shift_s=shift)
    assert detector.predict(data).fake_prob > 0.8


def test_late_mouth_reports_the_lag(detector: LipSyncDetector) -> None:
    score = detector.predict(LipSyncInput(frames=make_frames(lag_s=0.3), audio=make_audio()))
    assert score.fake_prob > 0.8
    assert score.details["best_lag_s"] == pytest.approx(0.3, abs=0.05)  # mouth after sound: +


def test_someone_elses_audio_scores_high(detector: LipSyncDetector) -> None:
    data = LipSyncInput(frames=make_frames(seed=0), audio=make_audio(seed=7))
    assert detector.predict(data).fake_prob > 0.8


def test_short_overlap_lowers_confidence(detector: LipSyncDetector) -> None:
    long = detector.predict(LipSyncInput(frames=make_frames(4.0), audio=make_audio(4.0)))
    short = detector.predict(LipSyncInput(frames=make_frames(1.5), audio=make_audio(1.5)))
    assert short.confidence < long.confidence


# --- calibration -----------------------------------------------------------------------


def _array_clips(n: int = 8) -> list[tuple[FrameBatch, AudioFeatures]]:
    return [(make_frames(seed=i), make_audio(seed=i)) for i in range(n)]


def test_calibration_rows_are_self_supervised(detector: LipSyncDetector) -> None:
    rows, labels, groups = calibration_rows(_array_clips(4), detector, np.random.default_rng(0))
    assert len(rows) == len(labels) == len(groups) == 4 * 4  # 1 real + 2 shifted + 1 swapped
    assert labels.count(0) == 4
    assert sorted(set(groups)) == [0, 1, 2, 3]


def test_calibration_round_trip(detector: LipSyncDetector, tmp_path: Path) -> None:
    rows, labels, _ = calibration_rows(_array_clips(), detector, np.random.default_rng(0))
    model = fit_calibration(rows, labels)

    real = save_calibration(model, tmp_path / "real.joblib", data_source="data/raw/x")
    detector.load(real)
    assert detector.is_trained
    in_sync = detector.predict(LipSyncInput(frames=make_frames(), audio=make_audio()))
    shifted = detector.predict(
        LipSyncInput(frames=make_frames(), audio=make_audio(), audio_shift_s=0.6)
    )
    assert in_sync.trained and in_sync.fake_prob < 0.5 < shifted.fake_prob

    synthetic = save_calibration(model, tmp_path / "syn.joblib", data_source="synthetic")
    detector.load(synthetic)
    assert not detector.is_trained  # synthetic calibration never ends demo mode

    detector.load(tmp_path / "missing.joblib")
    assert not detector.is_trained


def test_calibration_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="both"):
        fit_calibration([{"sync_corr": 1, "peak_corr": 1, "abs_best_lag_s": 0,
                          "peak_minus_median": 1}], [0])  # fmt: skip


def test_calibration_with_other_features_is_refused(
    detector: LipSyncDetector, tmp_path: Path
) -> None:
    import joblib

    path = tmp_path / "old.joblib"
    joblib.dump({"model": None, "feature_names": ("a", "b")}, path)
    with pytest.raises(ValueError, match="different features"):
        detector.load(path)


# --- end to end on generated talking-face videos ---------------------------------


@pytest.fixture(scope="module")
def extractor() -> Iterator[FrameExtractor]:
    if not all(assets_status().values()):
        pytest.skip("run: uv run python scripts/download_assets.py")
    with FrameExtractor(Settings(_env_file=None)) as fx:
        yield fx


@needs_mediapipe_assets
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_real_pipeline_separates_sync_from_lag(
    extractor: FrameExtractor, detector: LipSyncDetector, tmp_path: Path
) -> None:
    """Video -> FrameExtractor + AudioExtractor -> detector, on speech-rhythm clips."""
    audio_x = AudioExtractor(Settings(_env_file=None))

    def score(lag: float):
        clip = write_talking_face_clip(tmp_path / f"c{lag}.mp4", 4.0, lag_s=lag, rhythm="speech")
        data = LipSyncInput(frames=extractor.extract(clip), audio=audio_x.extract(clip))
        return detector.predict(data)

    sync, late = score(0.0), score(0.5)
    assert sync.fake_prob < 0.1 and abs(sync.details["best_lag_s"]) <= 0.04
    assert late.fake_prob > 0.9 and late.details["best_lag_s"] == pytest.approx(0.5, abs=0.05)

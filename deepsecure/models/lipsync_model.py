"""Layer 3C: does the mouth move in time with the sound? (cross-correlation baseline)

Idea: when someone speaks, their mouth opens wider on louder syllables, so the
*mouth-opening* signal and the audio *energy* signal rise and fall together. In a
dubbed video or a badly synthesised deepfake they don't, or they do but shifted.

Steps:

1. **Common grid.** The mouth timeline (from ``FrameBatch``, ~15/s with uneven
   steps) and the audio energy (100/s, starting at ``AudioFeatures.offset_s``) are
   both resampled onto one even grid (``lipsync_grid_hz``, 25/s). The energy is first
   smoothed with a moving average one grid-step long: a low-pass filter, so the
   resampling doesn't alias (the same Nyquist idea as in Phase 3).
2. **Cross-correlation.** For every lag L in +-``lipsync_max_lag_s`` we compute the
   Pearson correlation between ``mouth[t]`` and ``energy[t - L]`` over the points
   where both are known. Positive L = the mouth moves *after* the sound.
3. **Features** (all explainable):
   ``sync_corr`` = best correlation with |L| <= ``lipsync_sync_tolerance_s``
   (real speech: high), ``peak_corr`` / ``best_lag_s`` = best correlation anywhere
   and where, ``peak_minus_median`` = how much the peak stands out from the other
   lags (a flat curve means "no evidence either way").
4. **Score.** If a calibration file exists (``load()``), a logistic regression
   trained on real clips vs. the same clips with the audio shifted turns the
   features into ``fake_prob``. Otherwise a hand-set, *uncalibrated* rule is used
   and the score is flagged ``trained=False`` (demo mode)::

       evidence = (midpoint - sync_corr) + (peak_corr - sync_corr)
       fake_prob = sigmoid(slope * evidence)

   The first term says "mouth and sound don't match near zero lag"; the second
   says "they match much better at another offset" (dubbing, a delayed synthetic
   voice). The spec's original "max correlation over all lags" alone can't see an
   offset at all: a 0.5 s-late mouth still has a high peak (measured 0.93).

The detector refuses to judge (``available=False``) without audio, without a
mouth timeline (still images, no face), with too little overlap, or when the
mouth barely moves: correlation is meaningless if one signal is flat.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from deepsecure.config.settings import Settings, get_settings
from deepsecure.models.base import unavailable
from deepsecure.schemas import AudioFeatures, FrameBatch, LipSyncInput, ModalityScore

logger = logging.getLogger(__name__)

MOUTH_STATIC_REASON = "the mouth barely moves, so lip-sync can't be judged"
FEATURE_NAMES: tuple[str, ...] = ("sync_corr", "peak_corr", "abs_best_lag_s", "peak_minus_median")
CALIBRATION_FILE = "lipsync_baseline_calibration.joblib"


# --- signal helpers (pure functions, easy to test) ----------------------------


def even_grid(t_start: float, t_end: float, hz: float) -> np.ndarray:
    """Evenly spaced times from ``t_start`` to ``t_end`` (inclusive) at ``hz`` points/second."""
    if t_end < t_start:
        return np.zeros(0)
    n = int(np.floor((t_end - t_start) * hz + 1e-9)) + 1
    return t_start + np.arange(n) / hz


def mouth_on_grid(timestamps_s: list[float], mouth: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation of the mouth signal onto ``grid``; NaN where it isn't known.

    A grid point is unknown if the *nearest* original sample had no face (NaN), so
    gaps in face detection stay gaps instead of being bridged by a straight line.
    """
    t = np.asarray(timestamps_s, dtype=float)
    m = np.asarray(mouth, dtype=float)
    out = np.full(len(grid), np.nan)
    valid = np.isfinite(m)
    if valid.sum() < 2:
        return out
    out[:] = np.interp(grid, t[valid], m[valid])
    nearest = np.abs(grid[:, None] - t[None, :]).argmin(axis=1)
    out[~valid[nearest]] = np.nan
    out[(grid < t[0]) | (grid > t[-1])] = np.nan
    return out


def energy_on_grid(audio: AudioFeatures, grid: np.ndarray, shift_s: float = 0.0) -> np.ndarray:
    """Audio RMS energy on ``grid`` (video timeline), optionally played ``shift_s`` later.

    Energy frame k sits at ``offset_s + k * hop / sr`` seconds (librosa centres
    frames). Before ``offset_s`` and after the trimmed end the audio was trimmed as
    silence, so the energy there is 0. Outside the recording entirely it's NaN.
    """
    env = np.asarray(audio.energy_envelope, dtype=float)
    frames_per_s = audio.sample_rate / audio.hop_length
    grid_hz = 1.0 / np.median(np.diff(grid)) if len(grid) > 1 else frames_per_s
    width = max(1, int(round(frames_per_s / grid_hz)))
    smooth = np.convolve(env, np.ones(width) / width, mode="same")  # low-pass before resampling
    t_env = audio.offset_s + np.arange(len(env)) / frames_per_s
    t_orig = grid - shift_s  # what moment of the original recording plays at each grid time
    out = np.interp(t_orig, t_env, smooth, left=0.0, right=0.0)
    out[(t_orig < 0) | (t_orig > audio.duration_s)] = np.nan
    return out


def lag_correlations(
    mouth: np.ndarray, energy: np.ndarray, max_lag: int, min_overlap: int
) -> np.ndarray:
    """Pearson correlation of ``mouth[i]`` with ``energy[i - L]`` for L = -max_lag..max_lag.

    Returns an array of length ``2 * max_lag + 1`` (index ``max_lag`` is L = 0);
    NaN where fewer than ``min_overlap`` points overlap or a signal is constant.
    """
    n = len(mouth)
    out = np.full(2 * max_lag + 1, np.nan)
    for j, lag in enumerate(range(-max_lag, max_lag + 1)):
        lo, hi = max(0, lag), min(n, n + lag)
        if hi - lo < min_overlap:
            continue
        a, b = mouth[lo:hi], energy[lo - lag : hi - lag]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_overlap:
            continue
        a, b = a[ok], b[ok]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        out[j] = float(np.corrcoef(a, b)[0, 1])
    return out


def uncalibrated_fake_prob(feats: dict[str, float], settings: Settings) -> float:
    """The hand-set demo-mode rule (see the module docstring). Not fitted to any data."""
    offset_gap = feats["peak_corr"] - feats["sync_corr"]
    evidence = (settings.lipsync_corr_midpoint - feats["sync_corr"]) + offset_gap
    return float(1.0 / (1.0 + np.exp(-settings.lipsync_corr_slope * evidence)))


# --- the detector ---------------------------------------------------------------


class LipSyncDetector:
    """Cross-correlation lip-sync detector (implements :class:`BaseDetector`)."""

    name = "lipsync"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._calibration: Pipeline | None = None
        self.calibration_info: dict = {}

    @property
    def default_checkpoint(self) -> Path:
        return self.settings.models_dir / "lipsync" / CALIBRATION_FILE

    @property
    def is_trained(self) -> bool:
        """True only for a calibration fitted on real clips (synthetic ones are for demos)."""
        return (
            self._calibration is not None
            and self.calibration_info.get("data_source") != "synthetic"
        )

    def load(self, checkpoint: Path | None = None) -> None:
        """Load a calibration file if it exists; otherwise stay in (logged) demo mode.

        Only load files *you* created with ``scripts/calibrate_lipsync.py``: joblib
        files are pickles, and unpickling an untrusted file can run arbitrary code.
        """
        path = checkpoint or self.default_checkpoint
        if not path.is_file():
            logger.warning("No lip-sync calibration at %s: using the uncalibrated curve", path)
            self._calibration = None
            self.calibration_info = {}
            return
        bundle = joblib.load(path)
        if tuple(bundle.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError(f"{path} was made for different features; re-run the calibration")
        self._calibration = bundle["model"]
        self.calibration_info = {k: v for k, v in bundle.items() if k != "model"}
        logger.info("Loaded lip-sync calibration from %s (%s)", path, bundle.get("data_source"))

    # -- features ------------------------------------------------------------------

    def features(self, data: LipSyncInput) -> dict[str, float] | str:
        """The explainable sync features, or a string saying why they can't be computed."""
        s = self.settings
        frames, audio = data.frames, data.audio
        if audio is None:
            return "no usable audio"
        if frames is None or len(frames.timestamps_s) == 0:
            return "no mouth-movement timeline (still image or no video)"
        finite = np.isfinite(frames.mouth_open_seq)
        if finite.sum() < 2:
            return "no face/mouth found in the lip-sync window"

        grid = even_grid(frames.timestamps_s[0], frames.timestamps_s[-1], s.lipsync_grid_hz)
        mouth = mouth_on_grid(frames.timestamps_s, frames.mouth_open_seq, grid)
        energy = energy_on_grid(audio, grid, shift_s=data.audio_shift_s)
        both = np.isfinite(mouth) & np.isfinite(energy)
        valid_s = float(both.sum() / s.lipsync_grid_hz)
        if valid_s < s.lipsync_min_valid_s:
            return f"only {valid_s:.1f} s with both a visible mouth and audio"
        mouth_std = float(np.nanstd(mouth[both]))
        if mouth_std < s.lipsync_min_mouth_std:
            return MOUTH_STATIC_REASON

        max_lag = int(round(s.lipsync_max_lag_s * s.lipsync_grid_hz))
        min_overlap = max(3, int(s.lipsync_min_valid_s * s.lipsync_grid_hz) // 2)
        corrs = lag_correlations(mouth, energy, max_lag, min_overlap)
        if not np.isfinite(corrs).any():
            return "the audio energy is flat over the lip-sync window"
        lags_s = np.arange(-max_lag, max_lag + 1) / s.lipsync_grid_hz
        near_zero = np.abs(lags_s) <= s.lipsync_sync_tolerance_s + 1e-9
        best = int(np.nanargmax(corrs))
        return {
            "sync_corr": float(np.nanmax(np.where(near_zero, corrs, np.nan))),
            "peak_corr": float(corrs[best]),
            "best_lag_s": float(lags_s[best]),
            "abs_best_lag_s": float(abs(lags_s[best])),
            "peak_minus_median": float(corrs[best] - np.nanmedian(corrs)),
            "valid_s": valid_s,
            "mouth_std": mouth_std,
        }

    @staticmethod
    def feature_vector(features: dict[str, float]) -> np.ndarray:
        return np.array([[features[k] for k in FEATURE_NAMES]], dtype=float)

    # -- scoring ---------------------------------------------------------------------

    def predict(self, features: LipSyncInput) -> ModalityScore:
        """Score one clip. Never raises for missing evidence: returns ``available=False``."""
        start = time.perf_counter()
        feats = self.features(features)
        if isinstance(feats, str):
            return unavailable("lipsync", feats, (time.perf_counter() - start) * 1000)

        if self._calibration is not None:
            fake_prob = float(self._calibration.predict_proba(self.feature_vector(feats))[0, 1])
        else:
            fake_prob = uncalibrated_fake_prob(feats, self.settings)

        coverage = min(1.0, feats["valid_s"] / self.settings.lipsync_full_confidence_s)
        confidence = coverage * abs(2 * fake_prob - 1)
        return ModalityScore(
            name="lipsync",
            fake_prob=fake_prob,
            confidence=confidence,
            latency_ms=(time.perf_counter() - start) * 1000,
            trained=self.is_trained,
            details=feats,
        )


# --- calibration ("training" the baseline) ----------------------------------------------


def fit_calibration(rows: list[dict[str, float]], labels: list[int]) -> Pipeline:
    """Logistic regression on the sync features. ``labels``: 1 = fake (out of sync), 0 = real."""
    if len(set(labels)) < 2:
        raise ValueError("calibration needs both in-sync (0) and out-of-sync (1) examples")
    x = np.array([[r[k] for k in FEATURE_NAMES] for r in rows], dtype=float)
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    model.fit(x, np.asarray(labels))
    return model


def save_calibration(model: Pipeline, path: Path, **metadata: object) -> Path:
    """Save the model with its feature names and metadata (``data_source``, metrics...)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, **metadata}, path)
    return path


def calibration_rows(
    clips: list[tuple[FrameBatch, AudioFeatures]],
    detector: LipSyncDetector,
    rng: np.random.Generator,
    shifts_per_clip: int = 2,
    shift_range_s: tuple[float, float] = (0.3, 1.5),
    groups: list | None = None,
    swap_keys: list | None = None,
) -> tuple[list[dict[str, float]], list[int], list]:
    """Self-supervised training rows from *real* clips only. Returns (rows, labels, groups).

    For each real clip: the clip as-is -> label 0 (in sync); the same clip with its audio
    shifted by a random 0.3-1.5 s (either direction) -> label 1; its video with another
    clip's audio -> label 1.

    ``groups`` (one per clip, default: the clip index) is copied to every row of that clip, so
    a split by group keeps a clip and its negatives together; pass the *speaker* to keep
    each person on one side too. ``swap_keys`` (e.g. the sentence spoken) forces the swapped
    audio to come from a clip with a *different* key: two recordings of the same sentence can
    be nearly in sync, which would make a bad "out of sync" example.
    """
    groups = list(range(len(clips))) if groups is None else list(groups)
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    row_groups: list = []

    def add(frames: FrameBatch, audio: AudioFeatures, shift: float, label: int, i: int) -> None:
        feats = detector.features(LipSyncInput(frames=frames, audio=audio, audio_shift_s=shift))
        if isinstance(feats, dict):  # skip clips that can't be judged (and say so in the log)
            rows.append(feats)
            labels.append(label)
            row_groups.append(groups[i])
        else:
            logger.info("calibration: skipped a row of clip %d (%s)", i, feats)

    for i, (frames, audio) in enumerate(clips):
        add(frames, audio, 0.0, 0, i)
        for _ in range(shifts_per_clip):
            shift = float(rng.uniform(*shift_range_s)) * float(rng.choice([-1.0, 1.0]))
            add(frames, audio, shift, 1, i)
        partners = [
            k
            for k in range(len(clips))
            if k != i and (swap_keys is None or swap_keys[k] != swap_keys[i])
        ]
        if partners:
            add(frames, clips[int(rng.choice(partners))][1], 0.0, 1, i)
    return rows, labels, row_groups

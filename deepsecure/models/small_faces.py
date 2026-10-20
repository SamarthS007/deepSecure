"""Scoring the small faces of a group photo (used by ``visual_model``; DOCUMENTATION §8.18).

The face model was trained on close-up faces. A face that is 80 px wide in a group photo
is blurry once enlarged to 299 x 299, and genuine faces then score higher (measured on
DeepFakeFace validation faces: 20 % cross 0.40 at 80 px, 4 % at full size), although the
model still ranks real below fake (AUC 0.96). Two corrections:

1. **Size.** Genuine validation faces are shrunk (from their original pixels) to each size
   in ``source_px`` and enlarged to 299 as the app does; their fake logits are summarised
   by a mean and standard deviation per size. A face's
   ``z = (logit p - mu(size)) / sigma(size)`` says how unusual it is *for a genuine face
   of that size* (sizes between grid points are interpolated on a log scale; beyond the
   grid the end values are used).
2. **Number of faces.** Checking N faces gives N chances of a false alarm, so the most
   unusual face is judged with the Šidák correction: ``u = Phi(z_max) ** N`` is the chance
   that none of N genuine faces would look that unusual.

Finally ``u`` is mapped onto the scores of *full-size* genuine faces,
``p = sigmoid(mu_ref + sigma_ref * Phi^-1(u))``: a genuine group photo then scores like one
genuine portrait, so the usual risk thresholds and confidence rule apply unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import norm

SMALL_FACE_FILE = "small_face_calibration.json"
DEFAULT_SIZES = (48, 64, 80, 96, 112, 128)  # source sides (px) simulated when fitting


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def shrink(region: np.ndarray, source_px: int, size: int = 299) -> np.ndarray:
    """The app's crop of this face region if it had been only ``source_px`` pixels wide.

    ``region`` is the face's square region at its own resolution: it is shrunk once and
    enlarged once to ``size``, exactly like a small face in a group photo (shrinking an
    already enlarged crop would add blur the app never sees). Sizes at or above the
    region's own just give the app's normal crop.
    """
    small = region
    if source_px < region.shape[0]:
        small = cv2.resize(region, (source_px, source_px), interpolation=cv2.INTER_AREA)
    interp = cv2.INTER_AREA if small.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(small, (size, size), interpolation=interp)


def fit_table(
    real_probs: dict[int, np.ndarray], reference: np.ndarray, reference_px: float, **meta
) -> dict:
    """Calibration table from genuine faces: per size, mean and std of the fake logit.

    ``real_probs[size]`` are genuine faces shrunk to ``size``; ``reference`` the same faces
    at their own resolution (median source side ``reference_px``), which is also the last
    grid point. Simulated sizes at or above ``reference_px`` are left out.
    """
    sizes = sorted(s for s in real_probs if s < reference_px)
    if not sizes:
        raise ValueError("no simulated size below the reference size")
    stats = [(float(logit(real_probs[s]).mean()), float(logit(real_probs[s]).std(ddof=1)))
             for s in sizes]  # fmt: skip
    ref = (float(logit(reference).mean()), float(logit(reference).std(ddof=1)))
    return {
        "source_px": [float(s) for s in sizes] + [float(reference_px)],
        "mu": [m for m, _ in stats] + [ref[0]],
        "sigma": [s for _, s in stats] + [ref[1]],
        "reference": {"mu": ref[0], "sigma": ref[1]},
        "n_real": int(len(reference)),
        **meta,
    }


def small_face_z(probs: np.ndarray, source_px: np.ndarray, calib: dict) -> np.ndarray:
    """How unusual each face is compared with genuine faces of the same source size."""
    sizes = np.log(np.asarray(calib["source_px"], np.float64))
    w = np.log(np.maximum(np.asarray(source_px, np.float64), 1.0))
    mu = np.interp(w, sizes, calib["mu"])  # clamped at both ends
    sigma = np.interp(w, sizes, calib["sigma"])
    return (logit(probs) - mu) / sigma


def group_fake_prob(probs: np.ndarray, source_px: np.ndarray, calib: dict) -> tuple[float, int]:
    """(calibrated fake probability of the photo, index of the most unusual face)."""
    z = small_face_z(probs, source_px, calib)
    worst = int(np.argmax(z))
    # 1 - u, in log space so that very unusual faces don't round to u = 1
    tail = float(-np.expm1(len(z) * norm.logcdf(z[worst])))
    ref = calib["reference"]
    value = ref["mu"] + ref["sigma"] * float(norm.isf(np.clip(tail, 1e-300, 1.0)))
    return float(1 / (1 + np.exp(-value))), worst

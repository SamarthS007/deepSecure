"""Evaluation metrics shared by training scripts and the Phase 4D evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold where it occurs. ``labels``: 1 = fake/spoof.

    The EER is the error rate at the threshold where the false-acceptance rate (fakes
    passed as real) equals the false-rejection rate (real flagged as fake). Lower is
    better; 0 = perfect, 0.5 = chance. It's the standard ASVspoof metric because it
    needs no decision threshold to be chosen in advance.
    """
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        raise ValueError("EER needs both classes")
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    # thresholds[i] is an observed score; any cut between it and the next lower score
    # gives the same errors on this data. The midpoint leaves a margin on both sides,
    # which matters when the data is (nearly) separable and the "gap" is wide.
    upper = min(float(thresholds[i]), 1.0)  # roc_curve puts +inf first
    lower = float(thresholds[i + 1]) if i + 1 < len(thresholds) else upper
    return eer, (upper + lower) / 2.0


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Everything Phase 4D reports for one detector on one split. ``labels``: 1 = fake.

    Threshold-based: accuracy, precision (of the files flagged fake, how many were fake),
    recall (of the fakes, how many were flagged), F1 (their harmonic mean) and the
    confusion-matrix counts. Threshold-free: ROC-AUC and EER (NaN if one class is missing).
    """
    y = np.asarray(labels).astype(int)
    p = np.asarray(scores, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(set(y.tolist())) == 2
    eer, eer_thr = compute_eer(y, p) if both else (float("nan"), float("nan"))
    return {
        "n": int(len(y)),
        "n_fake": int(y.sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "eer": eer,
        "eer_threshold": eer_thr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }  # fmt: skip

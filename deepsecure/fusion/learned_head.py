"""Optional learned fusion: a logistic regression on top of the three detector scores.

Features per modality (6 in total): the score's **log-odds** ``log(p / (1 - p))``
(0 when unavailable) and an **available** flag (1/0). Log-odds put "0.99 vs 0.999"
and "0.5 vs 0.6" on a comparable scale, and the flag lets the model learn how much
to trust the others when one modality is missing, instead of pretending the
missing score was 0.5. Because the model is fitted on *detector outputs* (a second
level on top of the detectors), this is called **stacking**.

Honest limitation: fitting it needs files that have *all three* scores and a real/
fake label. FaceForensics++ (no audio) and ASVspoof (no video) can't provide that,
so the only source in this project is the internal set, which is small; the
training script therefore judges it by grouped cross-validation against the
weighted baseline and only saves it on request.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from deepsecure.schemas import ModalityScore

logger = logging.getLogger(__name__)

MODALITIES: tuple[str, ...] = ("visual", "audio", "lipsync")
FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{m}_{kind}" for m in MODALITIES for kind in ("logit", "available")
)
HEAD_FILE = "fusion_head.joblib"
EPS = 1e-4


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return float(np.log(p / (1 - p)))


def features_from_probs(probs: dict[str, float | None]) -> np.ndarray:
    """6 features from ``{modality: fake_prob or None/NaN when unavailable}``."""
    row: list[float] = []
    for m in MODALITIES:
        p = probs.get(m)
        ok = p is not None and not np.isnan(p)
        row += [_logit(p) if ok else 0.0, 1.0 if ok else 0.0]
    return np.array(row)


def features_from_frame(df: pd.DataFrame) -> np.ndarray:
    """Features from a table with ``visual``, ``audio``, ``lipsync`` columns (NaN = missing)."""
    cols = {m: df[m] if m in df else pd.Series(np.nan, index=df.index) for m in MODALITIES}
    return np.stack([
        features_from_probs({m: float(cols[m].iloc[i]) for m in MODALITIES})
        for i in range(len(df))
    ])  # fmt: skip


class LearnedFusionHead:
    """Fit / save / load / apply the stacking model."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.info: dict = {}

    @property
    def is_trained(self) -> bool:
        return self.model is not None and self.info.get("data_source") != "synthetic"

    def fit(self, df: pd.DataFrame, c: float = 1.0) -> LearnedFusionHead:
        y = df["label"].to_numpy().astype(int)
        if len(set(y.tolist())) < 2:
            raise ValueError("fusion training needs both real and fake examples")
        self.model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
        self.model.fit(features_from_frame(df), y)
        return self

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(features_from_frame(df))[:, 1]

    def predict_proba(self, scores: list[ModalityScore]) -> float:
        probs = {sc.name: sc.fake_prob for sc in scores if sc.available}
        return float(self.model.predict_proba(features_from_probs(probs)[None])[0, 1])

    def modality_weights(self) -> dict[str, float]:
        """Relative size of each modality's log-odds coefficient (sums to 1), for display."""
        coefs = dict(zip(FEATURE_NAMES, self.model.coef_[0], strict=True))
        mags = {m: abs(coefs[f"{m}_logit"]) for m in MODALITIES}
        total = sum(mags.values()) or 1.0
        return {m: v / total for m, v in mags.items()}

    def save(self, path: Path, **metadata: object) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": FEATURE_NAMES, **metadata}, path)
        return path

    def load(self, path: Path) -> bool:
        """Load a head you fitted yourself (joblib = pickle). Returns False if missing."""
        if not path.is_file():
            logger.info("No fusion head at %s", path)
            self.model, self.info = None, {}
            return False
        bundle = joblib.load(path)
        if tuple(bundle.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError(f"{path} was made for different features; re-run train_fusion.py")
        self.model = bundle["model"]
        self.info = {k: v for k, v in bundle.items() if k != "model"}
        return True

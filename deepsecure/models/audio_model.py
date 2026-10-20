"""Layer 3B: does the voice sound synthetic or cloned? (classic ML on audio statistics)

Input: ``AudioFeatures.summary_vector``, the 170 numbers from Phase 3 (mean and
std over time of 40 MFCCs, their deltas, and 5 spectral statistics), computed on
audio whose leading/trailing silence was **trimmed**. That matters: in ASVspoof
2019 LA, genuine recordings have longer silences than spoofs, and silence length
alone reaches up to 85 % accuracy (Müller et al., 2021). A model trained on
untrimmed audio can learn that shortcut instead of the voice.

Two candidate models, both ``StandardScaler`` -> classifier (scaling puts all 170
features on the same scale, which both models need):

* **Logistic regression**: a weighted sum of the features -> sigmoid. Fast,
  explainable, a strong baseline.
* **MLP** (multi-layer perceptron, hidden layers 256 and 64): can learn
  non-linear combinations; uses early stopping on an internal 10 % split.

Both are trained with *balanced sample weights* (ASVspoof train has ~9 spoofs per
bona fide utterance). The one with the lower **EER on the dev (val) split** is kept.

Demo mode: without ``models/audio/audio_classifier.joblib`` the detector returns
a neutral 0.5 with confidence 0 and ``trained=False``: there's nothing honest to say.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from deepsecure.config.settings import Settings, get_settings
from deepsecure.exceptions import DeepSecureError
from deepsecure.models.base import unavailable
from deepsecure.models.metrics import compute_eer
from deepsecure.preprocessing.audio_extractor import AudioExtractor, summary_feature_names
from deepsecure.schemas import AudioFeatures, ModalityScore

logger = logging.getLogger(__name__)

CLASSIFIER_FILE = "audio_classifier.joblib"


def candidate_models(settings: Settings, seed: int = 42) -> dict[str, Pipeline]:
    """The models compared on the dev split (fresh, unfitted)."""
    return {
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(C=settings.audio_logreg_c, max_iter=5000)
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=settings.audio_mlp_hidden,
                alpha=settings.audio_mlp_alpha,
                early_stopping=True,
                max_iter=300,
                random_state=seed,
            ),
        ),
    }


def fit_balanced(model: Pipeline, x: np.ndarray, y: np.ndarray) -> Pipeline:
    """Fit with per-sample weights so both classes count equally in the loss."""
    weights = compute_sample_weight("balanced", y)
    step = model.steps[-1][0]  # e.g. "logisticregression"
    return model.fit(x, y, **{f"{step}__sample_weight": weights})


def evaluate(model: Pipeline, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """EER (and its threshold), ROC-AUC and accuracy at 0.5 on one split."""
    p = model.predict_proba(x)[:, 1]
    eer, threshold = compute_eer(y, p)
    return {
        "eer": eer,
        "eer_threshold": threshold,
        "roc_auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
    }


def select_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    settings: Settings,
    seed: int = 42,
) -> tuple[str, Pipeline, dict[str, dict[str, float]]]:
    """Fit every candidate on train; return (best name, best model, all val results)."""
    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in candidate_models(settings, seed).items():
        start = time.perf_counter()
        fitted[name] = fit_balanced(model, x_train, y_train)
        results[name] = {
            **evaluate(fitted[name], x_val, y_val),
            "fit_seconds": time.perf_counter() - start,
        }
        logger.info("audio candidate %s: %s", name, results[name])
    best = min(results, key=lambda n: results[n]["eer"])
    return best, fitted[best], results


def save_classifier(model: Pipeline, path: Path, settings: Settings, **metadata: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_names": summary_feature_names(settings.n_mfcc),
        "n_mfcc": settings.n_mfcc,
        **metadata,
    }
    joblib.dump(bundle, path)
    return path


# --- feature extraction for whole splits (training and evaluation) ----------------------


def _summary(path: Path, settings: Settings) -> np.ndarray | None:
    try:
        return AudioExtractor(settings).extract(path).summary_vector
    except DeepSecureError as exc:
        print(f"[skip] {path.name}: {getattr(exc, 'reason', exc)}")
        return None


def summary_features(
    df: pd.DataFrame,
    split: str,
    s: Settings,
    n_jobs: int,
    cache_dir: Path,
    prefix: str = "asvspoof",
) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) for one split, from the cache when the same file list was extracted before."""
    part = df[df["split"] == split].reset_index(drop=True)
    digest = hashlib.sha1("\n".join(map(str, part["path"])).encode()).hexdigest()[:10]
    cache = cache_dir / f"{prefix}_{split}_{digest}_mfcc{s.n_mfcc}.npz"
    if cache.is_file():
        data = np.load(cache)
        print(f"[info] {split}: {len(data['y'])} cached feature vectors ({cache.name})")
        return data["x"], data["y"]
    start = time.perf_counter()
    vectors = Parallel(n_jobs=n_jobs)(delayed(_summary)(s.data_dir / p, s) for p in part["path"])
    keep = [i for i, v in enumerate(vectors) if v is not None]
    if not keep:
        raise ValueError(f"no usable audio files in split '{split}'")
    x = np.stack([vectors[i] for i in keep]).astype(np.float64)
    y = part["label"].to_numpy()[keep]
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, y=y)
    print(
        f"[info] {split}: extracted {len(keep)}/{len(part)} files in "
        f"{time.perf_counter() - start:.0f} s"
    )
    return x, y


class AudioDetector:
    """Spoofed-speech detector (implements :class:`BaseDetector`).

    Two backends (``settings.audio_backend``):

    * ``"aasist"`` (default): the pretrained AASIST network on the waveform
      (``aasist_model.py``); its weights come from the AASIST authors, trained on
      ASVspoof 2019 LA.
    * ``"classic"``: our logistic regression / MLP on the 170 MFCC statistics
      (``scripts/train_audio.py``).

    If the chosen backend's file is missing, the other one is used; if neither exists,
    the detector is in demo mode (0.5, confidence 0).
    """

    name = "audio"

    def __init__(self, settings: Settings | None = None, device=None) -> None:  # noqa: ANN001
        self.settings = settings or get_settings()
        self.device = device
        self.model: Pipeline | None = None
        self.aasist = None
        self.info: dict = {}

    @property
    def default_checkpoint(self) -> Path:
        return self.settings.models_dir / "audio" / CLASSIFIER_FILE

    @property
    def aasist_weights(self) -> Path:
        from deepsecure.models.aasist_model import WEIGHTS_FILE

        return self.settings.models_dir / "audio" / "aasist" / WEIGHTS_FILE

    @property
    def backend(self) -> str:
        return (
            "aasist" if self.aasist is not None else "classic" if self.model is not None else "demo"
        )

    @property
    def is_trained(self) -> bool:
        if self.backend == "demo":
            return False
        return self.info.get("data_source") != "synthetic"

    def load(self, checkpoint: Path | None = None) -> None:
        """Load the configured backend (falling back to the other one).

        ``checkpoint`` forces the classic classifier from that file (used by scripts and
        tests). joblib files are pickles: only load classifiers you trained yourself.
        """
        self.model, self.aasist, self.info = None, None, {}
        if checkpoint is None and self.settings.audio_backend == "aasist" and self._load_aasist():
            return
        if self.load_classic(checkpoint or self.default_checkpoint):
            return
        if checkpoint is None and self.settings.audio_backend == "classic" and self._load_aasist():
            return
        logger.warning("No audio model found (AASIST or classic classifier): demo mode")

    def _load_aasist(self) -> bool:
        from deepsecure.config.device import get_device
        from deepsecure.models.aasist_model import AASIST, DATA_SOURCE, FINETUNED_FILE, FINETUNED_META

        folder = self.aasist_weights.parent
        tuned, tuned_meta = folder / FINETUNED_FILE, folder / FINETUNED_META
        variant = self.settings.aasist_variant
        device = self.device or get_device(self.settings.device)
        if variant != "pretrained" and tuned.is_file() and tuned_meta.is_file():
            meta = json.loads(tuned_meta.read_text())
            self.aasist = AASIST(tuned, device, sha256=meta["sha256"])
            self.info = {
                "model_name": "AASIST (fine-tuned)",
                "data_source": meta["data_source"],
                "val_eer_threshold": float(meta["threshold"]),
                **{k: meta[k] for k in ("val_eer", "val_eer_by_set") if k in meta},
            }
            if meta.get("input", "raw") != self.settings.aasist_input:
                logger.warning("fine-tuned AASIST was trained on %s input", meta.get("input"))
            logger.info("Loaded fine-tuned AASIST on %s", device)
            return True
        if variant == "finetuned":
            logger.warning("No fine-tuned AASIST at %s (scripts/train_aasist.py)", tuned)
            return False
        if not self.aasist_weights.is_file():
            logger.info("No AASIST weights at %s", self.aasist_weights)
            return False
        self.aasist = AASIST(self.aasist_weights, device)
        self.info = {
            "model_name": "AASIST",
            "data_source": DATA_SOURCE,
            "val_eer_threshold": self.settings.aasist_threshold,
        }
        calib = folder / "calibration.json"  # from scripts/evaluate_audio.py
        if calib.is_file():
            meta = json.loads(calib.read_text())
            if meta.get("input") == self.settings.aasist_input:
                self.info["val_eer_threshold"] = float(meta["dev_eer_threshold"])
                self.info["dev_eer"] = meta.get("dev_eer")
                self.info["eval_eer"] = meta.get("eval_eer")
        logger.info("Loaded AASIST (%s input) on %s", self.settings.aasist_input, device)
        return True

    def load_classic(self, path: Path) -> bool:
        if not path.is_file():
            logger.info("No audio classifier at %s", path)
            return False
        bundle = joblib.load(path)
        if bundle.get("n_mfcc") != self.settings.n_mfcc:
            raise ValueError(
                f"{path} expects n_mfcc={bundle.get('n_mfcc')} but settings have "
                f"{self.settings.n_mfcc}; retrain with scripts/train_audio.py"
            )
        self.model = bundle["model"]
        self.info = {k: v for k, v in bundle.items() if k != "model"}
        logger.info(
            "Loaded audio classifier %s (%s, %s)",
            path,
            self.info.get("model_name"),
            self.info.get("data_source"),
        )
        return True

    def aasist_wave(self, features: AudioFeatures) -> np.ndarray:
        """The signal AASIST should see, per ``settings.aasist_input``."""
        if self.settings.aasist_input == "raw" and features.raw_waveform is not None:
            return features.raw_waveform
        return features.waveform

    def predict(self, features: AudioFeatures | None) -> ModalityScore:
        start = time.perf_counter()
        if features is None:
            return unavailable("audio", "no usable audio")
        voiced_s = len(features.waveform) / features.sample_rate
        details = {
            "voiced_s": voiced_s,
            "clipping_ratio": features.clipping_ratio,
            "silence_ratio": features.silence_ratio,
        }
        if self.backend == "demo":
            return ModalityScore(
                name="audio",
                fake_prob=0.5,
                confidence=0.0,
                trained=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                details={**details, "demo": 1.0},
            )
        if self.backend == "aasist":
            probs = self.aasist.window_probs(
                self.aasist_wave(features), self.settings.aasist_max_windows
            )
            fake_prob = float(probs.mean())
            details.update({"windows": float(len(probs)), "max_window_prob": float(probs.max())})
        else:
            x = features.summary_vector.reshape(1, -1).astype(np.float64)
            fake_prob = float(self.model.predict_proba(x)[0, 1])
        full_s = (
            self.settings.aasist_full_confidence_s
            if self.backend == "aasist"
            else self.settings.audio_full_confidence_s
        )
        coverage = min(1.0, voiced_s / full_s)
        threshold = float(self.info.get("val_eer_threshold", 0.5))
        return ModalityScore(
            name="audio",
            fake_prob=fake_prob,
            confidence=coverage * abs(2 * fake_prob - 1),
            latency_ms=(time.perf_counter() - start) * 1000,
            trained=self.is_trained,
            details={
                **details,
                "eer_threshold": threshold,
                "above_eer_threshold": float(fake_prob >= threshold),
            },
        )

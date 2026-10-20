"""Layer 3A: does the face show signs of manipulation? (CNN on face crops)

Model: **Xception** (``timm`` name ``legacy_xception``), a CNN pretrained on
ImageNet, with its 1000-class output layer replaced by a single output: the
*logit* of "fake". ``sigmoid(logit)`` is the fake probability of one crop.
Xception is the classic FaceForensics++ baseline, and its native input size
(299 x 299) is why Phase 3 crops faces to 299.

Inference on a :class:`FrameBatch`:

1. crops (N, 299, 299, 3) uint8 -> float tensor, normalised with the model's own
   mean/std (from ``timm.data.resolve_data_config``). The crops are *not*
   centre-cropped as timm's default evaluation transform would do: that would cut
   away the padded face border, where blending artifacts appear.
2. one forward pass per ``visual_batch_size`` crops -> N probabilities.
3. **video score** = mean of the ``visual_top_k`` (5) highest probabilities: a
   manipulation may be visible in only some frames, and a plain mean would dilute it.
4. **confidence** = agreement x decisiveness, where agreement = 1 - std/0.5 (crops
   disagree -> lower) and decisiveness = |2p - 1| (p near 0.5 -> lower). A single
   crop (a photo) has no spread, so only decisiveness counts.

**Group photos** (``FrameBatch.group_scan``, see ``frame_extractor``) are scored face by
face with a correction for small faces and for the number of faces (``small_faces``):
genuine group photos then score like one genuine portrait. The correction table
(``small_face_calibration.json``, from ``scripts/calibrate_small_faces.py``) belongs to one
checkpoint (SHA-256); without a matching table a group photo is ``unavailable``, not guessed.

Demo mode: without a checkpoint the network is built with *random* weights
(seeded, no download) so the pipeline can run end to end; every score is then
marked ``trained=False``. Checkpoints made from synthetic data never end demo mode.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn

from deepsecure.config.device import get_device
from deepsecure.config.settings import Settings, get_settings
from deepsecure.models.base import unavailable
from deepsecure.models.small_faces import SMALL_FACE_FILE, group_fake_prob
from deepsecure.schemas import FrameBatch, ModalityScore

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "visual_best.pt"


def build_model(arch: str, pretrained: bool, fallback: str | None = None) -> tuple[nn.Module, str]:
    """A timm CNN with one output logit. Returns ``(model, arch actually used)``."""
    try:
        return timm.create_model(arch, pretrained=pretrained, num_classes=1), arch
    except Exception as exc:  # unknown name, failed weight download...
        if fallback is None or fallback == arch:
            raise
        logger.warning("Could not build %s (%s); falling back to %s", arch, exc, fallback)
        return timm.create_model(fallback, pretrained=pretrained, num_classes=1), fallback


def input_config(model: nn.Module) -> dict:
    """The input size and normalisation the model expects (only these are used)."""
    cfg = timm.data.resolve_data_config({}, model=model)
    return {"input_size": int(cfg["input_size"][-1]), "mean": list(cfg["mean"]),
            "std": list(cfg["std"])}  # fmt: skip


def crops_to_tensor(
    crops: np.ndarray, input_size: int, mean: list[float], std: list[float]
) -> torch.Tensor:
    """(N, H, W, 3) uint8 RGB -> (N, 3, S, S) float, scaled to [0, 1] then normalised."""
    x = torch.from_numpy(np.ascontiguousarray(crops)).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != input_size or x.shape[-2] != input_size:
        x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", antialias=True)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s


def aggregate_probs(probs: np.ndarray, top_k: int) -> float:
    """Mean of the ``top_k`` largest crop probabilities (all of them if fewer)."""
    if len(probs) == 0:
        raise ValueError("no crop probabilities to aggregate")
    k = min(top_k, len(probs))
    return float(np.sort(probs)[-k:].mean())


def visual_confidence(probs: np.ndarray, fake_prob: float) -> float:
    """Agreement between crops x decisiveness of the final probability, in [0, 1]."""
    agreement = 1.0 - min(1.0, float(np.std(probs)) / 0.5) if len(probs) > 1 else 1.0
    return agreement * abs(2.0 * fake_prob - 1.0)


def save_checkpoint(model: nn.Module, arch: str, path: Path, **metadata: object) -> Path:
    """Weights + everything needed to rebuild and trust them (arch, input config, metrics)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "arch": arch,
        **input_config(model),
        "timm_version": timm.__version__,
        "created_at": datetime.now(UTC).isoformat(),
        **metadata,
    }
    torch.save(bundle, path)
    return path


class VisualDetector:
    """Xception face-crop classifier (implements :class:`BaseDetector`)."""

    name = "visual"

    def __init__(self, settings: Settings | None = None, device: torch.device | None = None):
        self.settings = settings or get_settings()
        self.device = device or get_device(self.settings.device)
        self.model: nn.Module | None = None
        self.arch = self.settings.visual_arch
        self.cfg: dict = {}
        self.checkpoint_info: dict = {}
        self._checkpoint_path: Path | None = None
        self._small_face: dict | None | bool = False  # False = not looked up yet

    @property
    def default_checkpoint(self) -> Path:
        return self.settings.models_dir / "visual" / CHECKPOINT_FILE

    @property
    def is_trained(self) -> bool:
        """True only for a checkpoint trained on real data (not synthetic, not missing)."""
        return bool(self.checkpoint_info) and self.checkpoint_info.get("data_source") != "synthetic"

    def load(self, checkpoint: Path | None = None) -> None:
        """Load trained weights, or build a random-weight model for demo mode.

        Checkpoints are read with ``torch.load(weights_only=True)``, which refuses to
        unpickle arbitrary Python objects, so a tampered ``.pt`` file can't run code.
        """
        path = checkpoint or self.default_checkpoint
        if path.is_file():
            bundle = torch.load(path, map_location="cpu", weights_only=True)
            model, arch = build_model(bundle["arch"], pretrained=False)
            model.load_state_dict(bundle["state_dict"])
            self.checkpoint_info = {k: v for k, v in bundle.items() if k != "state_dict"}
            self.cfg = {k: bundle[k] for k in ("input_size", "mean", "std")}
            self._checkpoint_path = path
            logger.info("Loaded visual checkpoint %s (%s, %s)", path, arch,
                        bundle.get("data_source"))  # fmt: skip
        else:
            logger.warning("No visual checkpoint at %s: demo mode (random weights)", path)
            torch.manual_seed(0)  # same demo scores on every run
            model, arch = build_model(
                self.settings.visual_arch, pretrained=False,
                fallback=self.settings.visual_fallback_arch,
            )  # fmt: skip
            self.checkpoint_info = {}
            self.cfg = input_config(model)
            self._checkpoint_path = None
        self._small_face = False
        self.arch = arch
        self.model = model.to(self.device).eval()

    def small_face_calibration(self) -> dict | None:
        """The small-face table for the loaded checkpoint, or None (missing / other model)."""
        if self._small_face is False:
            self._small_face = None
            path = self.settings.models_dir / "visual" / SMALL_FACE_FILE
            if self.is_trained and self._checkpoint_path is not None and path.is_file():
                calib = json.loads(path.read_text())
                digest = hashlib.sha256(self._checkpoint_path.read_bytes()).hexdigest()
                if calib.get("checkpoint_sha256") == digest:
                    self._small_face = calib
                else:
                    logger.warning("%s belongs to another checkpoint: ignored", path)
        return self._small_face

    @torch.inference_mode()
    def crop_probs(self, crops: np.ndarray) -> np.ndarray:
        """Fake probability of every crop, batch by batch."""
        if self.model is None:
            self.load()
        out: list[np.ndarray] = []
        bs = self.settings.visual_batch_size
        for i in range(0, len(crops), bs):
            x = crops_to_tensor(crops[i : i + bs], **self.cfg).to(self.device)
            out.append(torch.sigmoid(self.model(x)).squeeze(1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def predict(self, features: FrameBatch | None) -> ModalityScore:
        """Score one video or image from its face crops."""
        start = time.perf_counter()
        if features is None:
            return unavailable("visual", "no video or image frames")
        if features.faces_found == 0 or len(features.face_crops) == 0:
            return unavailable(
                "visual", f"no face found in {features.frames_sampled} sampled frame(s)",
                (time.perf_counter() - start) * 1000,
            )  # fmt: skip
        if features.group_scan:
            return self._predict_group(features, start)
        probs = self.crop_probs(features.face_crops)
        fake_prob = aggregate_probs(probs, self.settings.visual_top_k)
        return ModalityScore(
            name="visual",
            fake_prob=fake_prob,
            confidence=visual_confidence(probs, fake_prob),
            latency_ms=(time.perf_counter() - start) * 1000,
            trained=self.is_trained,
            details={
                "n_crops": float(len(probs)),
                "mean_prob": float(probs.mean()),
                "max_prob": float(probs.max()),
                "std_prob": float(probs.std()),
                "top_k": float(min(self.settings.visual_top_k, len(probs))),
                "face_found_ratio": features.face_found_ratio,
            },
        )

    def _predict_group(self, features: FrameBatch, start: float) -> ModalityScore:
        """A group photo: every face, corrected for its size and for the number of faces."""
        calib = self.small_face_calibration()
        if calib is None:
            return unavailable(
                "visual", f"group photo ({features.faces_found} small faces) needs the "
                "small-face calibration of this model: run scripts/calibrate_small_faces.py",
                (time.perf_counter() - start) * 1000,
            )  # fmt: skip
        probs = self.crop_probs(features.face_crops)
        sizes = np.asarray(features.crop_source_px, np.float64)
        fake_prob, worst = group_fake_prob(probs, sizes, calib)
        return ModalityScore(
            name="visual",
            fake_prob=fake_prob,
            confidence=visual_confidence(np.array([fake_prob]), fake_prob),
            latency_ms=(time.perf_counter() - start) * 1000,
            trained=self.is_trained,
            details={
                "n_crops": float(len(probs)),
                "group_photo": 1.0,
                "worst_face": float(worst),
                "worst_face_raw_prob": float(probs[worst]),
                "worst_face_px": float(sizes[worst]),
                "mean_prob": float(probs.mean()),
                "max_prob": float(probs.max()),
                "face_found_ratio": features.face_found_ratio,
            },
        )

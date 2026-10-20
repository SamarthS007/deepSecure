"""Training loop for the visual detector (fine-tuning Xception on face crops).

Recipe (each choice is explained in DOCUMENTATION §7.4):

* **Transfer learning**: start from ImageNet weights; epoch 1 trains only the new
  1-output head (backbone frozen) so the random head can't wreck the pretrained
  features with huge early gradients; then everything is fine-tuned.
* **Loss**: ``BCEWithLogitsLoss`` with ``pos_weight = n_real / n_fake``: FF++ has 4
  fake methods per real video, and without the weight the model could score
  well by leaning towards "fake".
* **Optimiser**: AdamW (lr 1e-4) with a cosine learning-rate schedule.
* **Augmentation** (train only): horizontal flip, colour jitter, random JPEG
  re-compression and blur, so the model can't rely on one dataset's exact
  compression, and copes better with photos and re-encoded uploads.
* **Model selection**: after every epoch, score the *validation* videos exactly as
  inference does (top-k mean over each video's crops) and keep the epoch with the
  best video-level ROC-AUC; stop after ``patience`` epochs without improvement.
* Mixed precision only on CUDA (MPS/CPU run in float32).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from deepsecure.models.visual_model import aggregate_probs, input_config, save_checkpoint

logger = logging.getLogger(__name__)


def build_transforms(input_size: int, mean: list[float], std: list[float], train: bool):
    """PIL image -> normalised float tensor; random augmentation only when ``train``."""
    steps: list = [v2.PILToTensor()]
    if train:
        steps += [
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(0.2, 0.2, 0.2, 0.02)], p=0.5),
            v2.RandomApply([v2.JPEG(quality=(40, 95))], p=0.5),
            v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        ]
    steps += [
        v2.Resize((input_size, input_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ]
    return v2.Compose(steps)


class FaceCropDataset(Dataset):
    """Face-crop images listed in a DataFrame with ``path`` and ``label`` (1 = fake)."""

    def __init__(self, df: pd.DataFrame, transform: Callable, root: Path | None = None):
        self.paths = [Path(p) if root is None else root / p for p in df["path"]]
        self.labels = df["label"].astype(np.float32).to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[i]) as img:
            x = self.transform(img.convert("RGB"))
        return x, torch.tensor([self.labels[i]])


@dataclass
class TrainResult:
    best_epoch: int
    best_val_auc: float
    history: list[dict] = field(default_factory=list)
    checkpoint: Path | None = None


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    head = set(map(id, model.get_classifier().parameters()))
    for p in model.parameters():
        if id(p) not in head:
            p.requires_grad = trainable


@torch.inference_mode()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(x.to(device))).squeeze(1).float().cpu().numpy() for x, _ in loader]
    return np.concatenate(out) if out else np.zeros(0)


def video_level(df: pd.DataFrame, crop_probs: np.ndarray, top_k: int) -> tuple[np.ndarray, ...]:
    """Aggregate crop probabilities per video exactly like inference. Returns (labels, scores)."""
    frame = df.assign(prob=crop_probs)
    grouped = frame.groupby("video", sort=True)
    labels = grouped["label"].first().to_numpy()
    scores = grouped["prob"].apply(lambda p: aggregate_probs(p.to_numpy(), top_k)).to_numpy()
    return labels, scores


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC, or NaN when only one class is present (AUC is undefined then)."""
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")


def train_visual(
    model: nn.Module,
    arch: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
    out_path: Path,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    freeze_epochs: int,
    patience: int,
    batch_size: int,
    top_k: int,
    root: Path | None = None,
    num_workers: int = 0,
    metadata: dict | None = None,
) -> TrainResult:
    """Fine-tune ``model`` and save the best epoch (by validation video ROC-AUC) to ``out_path``.

    ``train_df`` / ``val_df`` need ``path``, ``label`` (1 = fake) and ``video`` columns.
    """
    cfg = input_config(model)
    train_ds = FaceCropDataset(train_df, build_transforms(**cfg, train=True), root)
    val_ds = FaceCropDataset(val_df, build_transforms(**cfg, train=False), root)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers)

    n_fake = int(train_df["label"].sum())
    n_real = len(train_df) - n_fake
    if n_fake == 0 or n_real == 0:
        raise ValueError("training data must contain both real and fake crops")
    pos_weight = torch.tensor([n_real / n_fake], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    result = TrainResult(best_epoch=-1, best_val_auc=-math.inf)
    stale = 0
    for epoch in range(epochs):
        _set_backbone_trainable(model, trainable=epoch >= freeze_epochs)
        model.train()
        start, total, seen = time.perf_counter(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(x), y)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            total += float(loss.detach()) * len(x)
            seen += len(x)
        scheduler.step()

        val_probs = predict_loader(model, val_dl, device)
        y_val = val_df["label"].to_numpy()
        p_val = np.clip(val_probs, 1e-6, 1 - 1e-6)
        val_loss = float(-np.mean(y_val * np.log(p_val) + (1 - y_val) * np.log(1 - p_val)))
        v_labels, v_scores = video_level(val_df, val_probs, top_k)
        row = {
            "epoch": epoch + 1,
            "frozen_backbone": epoch < freeze_epochs,
            "train_loss": total / max(seen, 1),
            "val_loss": val_loss,
            "val_crop_auc": safe_auc(y_val, val_probs),
            "val_video_auc": safe_auc(v_labels, v_scores),
            "val_video_acc": float(np.mean((v_scores >= 0.5) == v_labels)),
            "seconds": time.perf_counter() - start,
        }
        result.history.append(row)
        logger.info("epoch %d: %s", epoch + 1, {k: round(v, 4) for k, v in row.items()
                                                if isinstance(v, float)})  # fmt: skip

        # Select on video AUC; if it's undefined (one class in val) fall back to val loss.
        score = row["val_video_auc"] if not math.isnan(row["val_video_auc"]) else -val_loss
        if score > result.best_val_auc:
            result.best_epoch, result.best_val_auc, stale = epoch + 1, score, 0
            result.checkpoint = save_checkpoint(
                model, arch, out_path, best_epoch=epoch + 1,
                val_video_auc=row["val_video_auc"], val_video_acc=row["val_video_acc"],
                val_crop_auc=row["val_crop_auc"], **(metadata or {}),
            )  # fmt: skip
        else:
            stale += 1
            if stale >= patience:
                logger.info("early stopping after epoch %d", epoch + 1)
                break
    return result

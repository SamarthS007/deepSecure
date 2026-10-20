"""Fine-tune AASIST on In-the-Wild + ASVspoof so that it stops equating "studio audio" with "real".

Why: the published AASIST learned ASVspoof's studio-recorded speech as genuine, so genuine
internet audio looks "spoofed" to it (DOCUMENTATION §8.16: AUC 0.295, 95 % false alarms on
real YouTube speech). In-the-Wild (Müller et al. 2022) has real *and* fake speech collected
from the same kind of online sources, so the label can't be read off the recording channel.

Recipe (start from the published weights; all of it on an 8 GB M1):

* **Balanced batches** from four pools: In-the-Wild real / fake (train speakers only) and
  ASVspoof real / fake (train partition), 25 % each, so the model keeps what it knew.
* **Augmentation** against channel and silence shortcuts, on every training crop:
  random silence trimming or padding, band-limiting (resample to 8-12 kHz and back),
  mu-law 8-bit quantisation (telephone-like), gain and additive noise.
* A random 64,600-sample crop per clip (the app averages windows over the whole clip).
* After each epoch: EER on In-the-Wild validation speakers and ASVspoof dev (app-style
  window averaging); the epoch with the lowest mean EER is kept. The decision threshold
  is the EER point of the pooled validation scores.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import nn

from deepsecure.models.aasist_model import NB_SAMP, windows
from deepsecure.models.metrics import compute_eer

logger = logging.getLogger(__name__)
SR = 16_000


def load_wave(path: str) -> np.ndarray:
    """Mono float32 at 16 kHz."""
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


def _trim(y: np.ndarray, rel: float = 0.02) -> np.ndarray:
    loud = np.flatnonzero(np.abs(y) > rel * (np.abs(y).max() + 1e-9))
    return y[loud[0] : loud[-1] + 1] if len(loud) else y


def augment(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Channel / silence augmentation (each step independently, with its own probability)."""
    y = y.astype(np.float32, copy=True)
    r = rng.random(6)
    if r[0] < 0.5:  # silence: remove it, or add a random amount, so its length means nothing
        y = _trim(y)
    elif r[0] < 0.75:
        pad = lambda: rng.normal(0, 1e-4, int(rng.uniform(0, 0.6) * SR)).astype(np.float32)  # noqa: E731
        y = np.concatenate([pad(), y, pad()])
    if r[1] < 0.4:  # band-limit: phone / low-rate recordings
        low = int(rng.choice([8000, 11025, 12000]))
        g = np.gcd(low, SR)
        y = resample_poly(resample_poly(y, low // g, SR // g), SR // g, low // g).astype(np.float32)
    if r[2] < 0.2:  # mu-law 8-bit (telephone codec)
        peak = np.abs(y).max() + 1e-9
        x = np.clip(y / peak, -1, 1)
        q = np.round((np.sign(x) * np.log1p(255 * np.abs(x)) / np.log(256) + 1) * 127.5)
        x = q / 127.5 - 1
        y = (np.sign(x) * ((256 ** np.abs(x)) - 1) / 255 * peak).astype(np.float32)
    if r[3] < 0.5:  # gain
        y = y * np.float32(10 ** (rng.uniform(-6, 6) / 20))
    if r[4] < 0.3:  # noise at 15-40 dB SNR
        p = float(np.mean(y**2)) + 1e-12
        y = y + rng.normal(0, np.sqrt(p / 10 ** (rng.uniform(15, 40) / 10)), len(y)).astype(
            np.float32
        )
    return np.clip(y, -1, 1).astype(np.float32)


def random_crop(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(y) == 0:
        y = np.zeros(SR, np.float32)
    if len(y) <= NB_SAMP:
        return np.tile(y, NB_SAMP // len(y) + 1)[:NB_SAMP]
    start = int(rng.integers(0, len(y) - NB_SAMP + 1))
    return y[start : start + NB_SAMP]


class BalancedClips(torch.utils.data.IterableDataset):
    """Endless stream of (crop, label), drawn equally from each pool of (path, label) lists."""

    def __init__(self, pools: list[list[tuple[str, int]]], seed: int, augment_p: float = 1.0):
        self.pools, self.seed, self.augment_p = pools, seed, augment_p

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 9973)
        while True:
            pool = self.pools[int(rng.integers(len(self.pools)))]
            path, label = pool[int(rng.integers(len(pool)))]
            try:
                y = load_wave(path)
            except (RuntimeError, sf.LibsndfileError):
                continue
            if rng.random() < self.augment_p:
                y = augment(y, rng)
            yield torch.from_numpy(random_crop(y, rng)), label


@torch.inference_mode()
def score_clips(
    model: nn.Module, paths: list[str], device: torch.device, max_windows: int = 8, batch: int = 4
) -> np.ndarray:
    """App-style P(fake) per clip: mean over its windows (raw waveform, no augmentation)."""
    model.eval()
    probs, buf, owners = [], [], []

    def flush():
        x = torch.from_numpy(np.stack(buf)).to(device)
        _, logits = model(x)
        p = torch.softmax(logits.float(), dim=1)[:, 0].cpu().numpy()
        for o, v in zip(owners, p, strict=True):
            probs[o].append(float(v))
        buf.clear()
        owners.clear()

    for i, path in enumerate(paths):
        probs.append([])
        for w in windows(load_wave(path), max_windows):
            buf.append(w)
            owners.append(i)
            if len(buf) == batch:
                flush()
    if buf:
        flush()
    return np.array([np.mean(p) for p in probs])


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    eer: dict[str, float]
    seconds: float


def train(
    model: nn.Module,
    pools: list[list[tuple[str, int]]],
    val_sets: dict[str, tuple[list[str], np.ndarray]],
    device: torch.device,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    seed: int,
    on_epoch=None,
    workers: int = 2,
    accum: int = 1,
) -> list[EpochResult]:
    """Fine-tune ``model`` in place; calls ``on_epoch(result, model, val_scores)`` per epoch.

    One step = ``accum`` mini-batches of ``batch_size`` (gradient accumulation: AASIST keeps
    full-resolution activations for backprop, so 8 GB of RAM fits only small batches).
    """
    torch.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        BalancedClips(pools, seed), batch_size=batch_size, num_workers=workers,
        persistent_workers=workers > 0,
    )  # fmt: skip
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss()
    history, it = [], iter(loader)
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.perf_counter(), []
        for step in range(steps_per_epoch):
            opt.zero_grad()
            step_loss = 0.0
            for _ in range(accum):
                x, y = next(it)
                # AASIST's output 1 = bona fide, so target class = 1 - fake
                _, logits = model(x.to(device))
                loss = loss_fn(logits, (1 - y).long().to(device)) / accum
                loss.backward()
                step_loss += loss.item()
            opt.step()
            sched.step()
            losses.append(step_loss)
            if (step + 1) % 100 == 0:
                logger.info("epoch %d step %d loss %.4f", epoch, step + 1, np.mean(losses[-100:]))
                print(f"[train] epoch {epoch} step {step + 1}/{steps_per_epoch} "
                      f"loss {np.mean(losses[-100:]):.4f}", flush=True)  # fmt: skip
        if device.type == "mps":
            torch.mps.empty_cache()
        val_scores = {
            name: score_clips(model, paths, device) for name, (paths, _) in val_sets.items()
        }
        eer = {
            name: float(compute_eer(labels, val_scores[name])[0])
            for name, (_, labels) in val_sets.items()
        }
        result = EpochResult(epoch, float(np.mean(losses)), eer, time.perf_counter() - t0)
        history.append(result)
        print(f"[val ] epoch {epoch}: " + ", ".join(f"{k} EER {v:.4f}" for k, v in eer.items())
              + f" ({result.seconds:.0f} s)", flush=True)  # fmt: skip
        if on_epoch:
            on_epoch(result, model, val_scores)
    return history

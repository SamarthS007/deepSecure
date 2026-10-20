"""AASIST: a pretrained spoofed-speech detector that reads the raw waveform.

AASIST (Jung et al., ICASSP 2022; code and weights by NAVER Corp., MIT licence) is a
graph-attention network over spectral and temporal features learnt directly from the
waveform. The authors trained it on ASVspoof 2019 LA (train partition) and report an
EER of 0.83 % on the LA evaluation set. We use their published weights unchanged:

* model code: ``deepsecure/models/third_party/aasist.py`` (vendored, unmodified);
* weights: ``models/audio/aasist/AASIST.pth`` (downloaded by ``download_assets.py``,
  pinned to a commit and checked against a SHA-256 hash, loaded with
  ``weights_only=True``).

Input: 64 600 samples (≈ 4 s at 16 kHz). Shorter audio is repeated to that length (as
in the authors' code); longer audio is cut into windows with 50 % overlap and the
fake probability is averaged over windows (the maximum is reported too). Output index
1 is "bona fide", so P(fake) = softmax(logits)[0].
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch

from deepsecure.models.third_party.aasist import Model as _AASIST

logger = logging.getLogger(__name__)

COMMIT = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"
WEIGHTS_URL = f"https://raw.githubusercontent.com/clovaai/aasist/{COMMIT}/models/weights/AASIST.pth"
WEIGHTS_SHA256 = "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"
WEIGHTS_FILE = "AASIST.pth"
# Our fine-tuned weights (scripts/train_aasist.py) and their metadata (SHA-256, threshold, data)
FINETUNED_FILE = "AASIST_finetuned.pth"
FINETUNED_META = "finetuned.json"
NB_SAMP = 64_600  # the authors' input length (~4.04 s at 16 kHz)
CONFIG = {  # config/AASIST.conf "model_config" at the pinned commit
    "architecture": "AASIST",
    "nb_samp": NB_SAMP,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}
DATA_SOURCE = "ASVspoof 2019 LA train (pretrained weights published by the AASIST authors)"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def windows(wave: np.ndarray, max_windows: int) -> np.ndarray:
    """(n, NB_SAMP) windows: tiled if short (authors' ``pad``), else 50 %-overlapping cuts."""
    wave = np.asarray(wave, dtype=np.float32)
    if len(wave) == 0:
        raise ValueError("empty waveform")
    if len(wave) <= NB_SAMP:
        reps = NB_SAMP // len(wave) + 1
        return np.tile(wave, reps)[:NB_SAMP][None]
    hop = NB_SAMP // 2
    starts = list(range(0, len(wave) - NB_SAMP + 1, hop))
    if starts[-1] + NB_SAMP < len(wave):
        starts.append(len(wave) - NB_SAMP)  # a last window ending at the end: no tail is skipped
    return np.stack([wave[s : s + NB_SAMP] for s in starts[:max_windows]])


class AASIST:
    """Loads the pretrained network once; scores waveforms (16 kHz mono float)."""

    def __init__(self, weights: Path, device: torch.device, sha256: str = WEIGHTS_SHA256) -> None:
        """``sha256``: the pinned published hash, or the one recorded when we fine-tuned."""
        if sha256_file(weights) != sha256:
            raise ValueError(f"{weights} does not match its recorded SHA-256; re-download/retrain")
        self.device = device
        self.model = _AASIST(CONFIG)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(device).eval()

    @torch.inference_mode()
    def window_probs(self, wave: np.ndarray, max_windows: int = 8, chunk: int = 2) -> np.ndarray:
        """P(fake) for every window of ``wave``, ``chunk`` windows per forward pass.

        AASIST keeps full-resolution feature maps (~0.5 GB per 4 s window on MPS), so all 8
        windows of a 30 s clip in one batch pushed an 8 GB Mac into swap; chunks cap it.
        """
        w = windows(wave, max_windows)
        out = []
        for i in range(0, len(w), chunk):
            _, logits = self.model(torch.from_numpy(w[i : i + chunk]).to(self.device))
            out.append(torch.softmax(logits.float(), dim=1)[:, 0].cpu().numpy())
        return np.concatenate(out)

    @torch.inference_mode()
    def batch_probs(self, waves: list[np.ndarray]) -> np.ndarray:
        """P(fake) for many short clips at once (first window of each); for evaluation."""
        x = torch.from_numpy(np.concatenate([windows(w, 1) for w in waves])).to(self.device)
        _, logits = self.model(x)
        return torch.softmax(logits.float(), dim=1)[:, 0].cpu().numpy()

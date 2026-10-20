"""Pick the compute device PyTorch should use.

Order of preference: NVIDIA GPU (``cuda``) -> Apple-silicon GPU (``mps``) ->
``cpu``. The choice can be forced with the ``DEEPSECURE_DEVICE`` setting.
"""

from __future__ import annotations

import logging

import torch

from deepsecure.config.settings import get_settings

logger = logging.getLogger(__name__)


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def get_device(preference: str | None = None) -> torch.device:
    """Return the best available ``torch.device`` and log which one was chosen.

    Args:
        preference: ``"auto"``, ``"cuda"``, ``"mps"`` or ``"cpu"``. Defaults to
            the ``device`` setting. If a specific device is requested but isn't
            available, we fall back to the CPU with a warning instead of crashing.

    Returns:
        The selected ``torch.device``.
    """
    pref = (preference or get_settings().device).lower()

    if pref == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif _mps_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif pref == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif pref == "mps" and _mps_available():
        device = torch.device("mps")
    elif pref == "cpu":
        device = torch.device("cpu")
    else:
        logger.warning("Requested device %r is not available; falling back to CPU.", pref)
        device = torch.device("cpu")

    logger.info("Using compute device: %s", device)
    return device


def describe_device(device: torch.device) -> str:
    """Human-readable name of the device, for reports and the UI."""
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"

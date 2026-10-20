"""The interface every detector (visual, audio, lip-sync) implements.

The pipeline only ever talks to this interface, so a detector can be swapped
(baseline -> learned model) without touching the pipeline. ``typing.Protocol``
means "anything with these methods counts": no inheritance needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from deepsecure.schemas import ModalityName, ModalityScore


@runtime_checkable
class BaseDetector(Protocol):
    """One evidence stream: features in, :class:`ModalityScore` out."""

    name: ModalityName

    def load(self, checkpoint: Path | None = None) -> None:
        """Load trained weights / calibration. ``None`` = the default path (may not exist yet)."""
        ...

    def predict(self, features: Any) -> ModalityScore:
        """Score one file. Must return ``available=False`` (not raise) when it can't judge."""
        ...

    @property
    def is_trained(self) -> bool:
        """False -> the score comes from untrained/uncalibrated code (demo mode)."""
        ...


def unavailable(name: ModalityName, reason: str, latency_ms: float = 0.0) -> ModalityScore:
    """A score that says "this evidence is missing"; fusion ignores it."""
    return ModalityScore(
        name=name,
        fake_prob=0.5,
        confidence=0.0,
        available=False,
        reason=reason,
        latency_ms=latency_ms,
    )

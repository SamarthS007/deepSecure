"""Custom exceptions, so callers can tell *our* expected failures apart from bugs."""

from __future__ import annotations


class DeepSecureError(Exception):
    """Base class for every error DeepSecure raises on purpose."""


class InvalidInputError(DeepSecureError):
    """The uploaded file can't be analysed (wrong type, too big, corrupted, too short...)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ModelNotTrainedError(DeepSecureError):
    """A component needs a trained checkpoint that doesn't exist yet."""


class AssetMissingError(DeepSecureError):
    """A required downloaded file (e.g. a MediaPipe model) is missing from disk."""


class ModalityUnavailableError(DeepSecureError):
    """One evidence stream can't be produced for this file (no audio track, silent audio...).

    Not a failure of the whole analysis: the pipeline marks that modality
    ``available=False`` with this ``reason`` and fusion ignores it.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

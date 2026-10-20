"""Logging setup: one console handler plus a rotating file in ``logs/``.

Call :func:`setup_logging` once at program start (the app, scripts and the
pipeline all do). Modules then just use ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from deepsecure.config.settings import get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURED_FLAG = "_deepsecure_configured"


def setup_logging(level: str | None = None, logs_dir: Path | None = None) -> logging.Logger:
    """Configure the ``deepsecure`` logger. Safe to call more than once.

    Args:
        level: Log level name. Defaults to the ``log_level`` setting.
        logs_dir: Folder for ``deepsecure.log``. Defaults to the ``logs_dir`` setting.

    Returns:
        The configured ``deepsecure`` package logger.
    """
    settings = get_settings()
    logger = logging.getLogger("deepsecure")
    logger.setLevel((level or settings.log_level).upper())

    # Streamlit re-runs the script on every interaction; without this guard we
    # would add a new pair of handlers each time and every line would repeat.
    if getattr(logger, _CONFIGURED_FLAG, False):
        return logger

    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_dir = logs_dir or settings.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "deepsecure.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    setattr(logger, _CONFIGURED_FLAG, True)
    return logger

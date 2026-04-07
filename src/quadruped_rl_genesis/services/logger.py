"""Root logging configuration for console and optional file output."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: Path | None = None, level: str = "INFO") -> None:
    """Configure root logging for console output and optional file capture.

    Args:
        log_file (Path | None): File that will also receive log messages. If
            ``None``, only stderr logging is configured.
        level (str): Logging level name such as ``"INFO"`` or ``"DEBUG"``.
            Invalid names fall back to ``INFO``.
    """
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger bound to the given module or component name.

    Args:
        name (str): Logger namespace, usually ``__name__``.

    Returns:
        logging.Logger: Logger that inherits the root configuration.
    """
    return logging.getLogger(name)

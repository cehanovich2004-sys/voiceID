"""Centralized logging setup for VoiceID."""

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str | int = "INFO") -> logging.Logger:
    """Configure standard-library logging and return the VoiceID logger."""

    numeric_level = _coerce_log_level(level)
    logging.basicConfig(level=numeric_level, format=_LOG_FORMAT, force=True)

    logger = logging.getLogger("voiceid")
    logger.setLevel(numeric_level)
    return logger


def _coerce_log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    normalized_level = level.upper()
    numeric_level = logging.getLevelName(normalized_level)
    if isinstance(numeric_level, int):
        return numeric_level

    msg = f"Unsupported log level: {level}"
    raise ValueError(msg)

"""Minimal application configuration for VoiceID."""

from dataclasses import dataclass
from typing import Literal

DEFAULT_APP_NAME = "VoiceID"
DEFAULT_APP_VERSION = "0.1.0"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Configuration values required by the Phase 1 project foundation."""

    app_name: str = DEFAULT_APP_NAME
    version: str = DEFAULT_APP_VERSION
    log_level: LogLevel = "INFO"
    debug: bool = False


def get_default_config() -> AppConfig:
    """Return the default application configuration."""

    return AppConfig()

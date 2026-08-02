"""Smoke tests for the Phase 1 project foundation."""

import logging

import voiceid
from voiceid.config import AppConfig, get_default_config
from voiceid.logging_config import configure_logging


def test_package_imports_and_exposes_version() -> None:
    assert isinstance(voiceid.__version__, str)
    assert voiceid.__version__


def test_default_config_can_be_created() -> None:
    config = get_default_config()

    assert isinstance(config, AppConfig)
    assert config.app_name == "VoiceID"
    assert config.version == voiceid.__version__
    assert config.log_level == "INFO"
    assert config.debug is False


def test_logging_configuration_returns_voiceid_logger() -> None:
    logger = configure_logging("DEBUG")

    assert logger.name == "voiceid"
    assert logger.level == logging.DEBUG

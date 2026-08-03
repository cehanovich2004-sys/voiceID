"""Smoke tests for the Phase 1 project foundation."""

import logging
from dataclasses import FrozenInstanceError

import pytest

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


def test_default_config_is_immutable() -> None:
    config = get_default_config()

    with pytest.raises(FrozenInstanceError):
        config.app_name = "Changed"


def test_logging_configuration_returns_voiceid_logger() -> None:
    logger = configure_logging("DEBUG")

    assert logger.name == "voiceid"
    assert logger.level == logging.DEBUG


def test_logging_configuration_does_not_duplicate_root_handlers() -> None:
    configure_logging()
    first_handler_count = len(logging.getLogger().handlers)

    configure_logging()
    second_handler_count = len(logging.getLogger().handlers)

    assert second_handler_count == first_handler_count


def test_logging_configuration_rejects_invalid_level() -> None:
    with pytest.raises(ValueError, match="Unsupported log level"):
        configure_logging("TRACE")

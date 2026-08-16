"""Local Telegram voice collection MVP for VoiceID research."""

from voiceid.telegram_bot.config import (
    DEFAULT_DATA_DIR,
    MAX_VOICE_FILE_SIZE_BYTES,
    MAX_VOICE_SECONDS,
    MIN_VOICE_SECONDS,
    REQUIRED_RECORDINGS,
    RESEARCH_PHRASES,
)
from voiceid.telegram_bot.manifest import export_manifest_for_calibration
from voiceid.telegram_bot.storage import VoiceCollectionStore

__all__ = [
    "DEFAULT_DATA_DIR",
    "MAX_VOICE_FILE_SIZE_BYTES",
    "MAX_VOICE_SECONDS",
    "MIN_VOICE_SECONDS",
    "REQUIRED_RECORDINGS",
    "RESEARCH_PHRASES",
    "VoiceCollectionStore",
    "export_manifest_for_calibration",
]

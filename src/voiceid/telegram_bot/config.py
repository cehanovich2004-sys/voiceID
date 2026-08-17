"""Configuration constants for the local Telegram collection MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Final

REQUIRED_RECORDINGS: Final = 6
MIN_VOICE_SECONDS: Final = 1
MAX_VOICE_SECONDS: Final = 60
MAX_VOICE_FILE_SIZE_BYTES: Final = 20 * 1024 * 1024
FFMPEG_TIMEOUT_SECONDS: Final = 30.0

DEFAULT_DATA_DIR: Final = Path.home() / ".local" / "share" / "voiceid" / "telegram_bot"
DEFAULT_MODEL_CACHE_DIR: Final = (
    Path.home() / ".cache" / "voiceid" / "speechbrain_ecapa"
)

RESEARCH_PHRASES: Final = (
    "Сегодня хорошая погода, и я проверяю запись своего голоса.",
    "Сегодня хорошая погода, и я проверяю запись своего голоса.",
    "Сегодня хорошая погода, и я проверяю запись своего голоса.",
    "Мой голос может звучать по-разному в разные дни.",
    "Система сравнивает особенности речи, а не содержание фразы.",
    "Эта запись используется только для исследовательского теста.",
)

START_TEXT: Final = (
    "VoiceID проводит исследовательский тест записи голоса. Голосовые сообщения "
    "передаются через Telegram и сохраняются локально оператором теста. Это не "
    "production-аутентификация и не решение о личности."
)

CONSENT_REQUIRED_TEXT: Final = "Сначала нужно согласиться с условиями /start."
GENERIC_ERROR_TEXT: Final = "Не удалось обработать запись. Повторите текущую фразу."
UNSUPPORTED_MESSAGE_TEXT: Final = "Пришлите именно голосовое сообщение Telegram."

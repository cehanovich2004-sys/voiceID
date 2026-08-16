"""Telegram update orchestration for local VoiceID voice collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, cast

from voiceid.telegram_bot.audio import AudioConversionError, convert_ogg_to_wav
from voiceid.telegram_bot.config import (
    CONSENT_REQUIRED_TEXT,
    GENERIC_ERROR_TEXT,
    MAX_VOICE_FILE_SIZE_BYTES,
    MAX_VOICE_SECONDS,
    MIN_VOICE_SECONDS,
    RESEARCH_PHRASES,
    START_TEXT,
    UNSUPPORTED_MESSAGE_TEXT,
)
from voiceid.telegram_bot.storage import (
    ReservedSample,
    SessionSnapshot,
    VoiceCollectionStore,
    VoiceCollectionStoreError,
)
from voiceid.telegram_bot.telegram_api import TelegramApiError, TelegramClient

_YES: Final = "consent_yes"
_NO: Final = "consent_no"


@dataclass(frozen=True, slots=True)
class BotRuntimeConfig:
    """Runtime limits for local Telegram voice collection."""

    ffmpeg_path: str = "ffmpeg"
    min_voice_seconds: int = MIN_VOICE_SECONDS
    max_voice_seconds: int = MAX_VOICE_SECONDS
    max_voice_file_size_bytes: int = MAX_VOICE_FILE_SIZE_BYTES


class TelegramVoiceCollectionBot:
    """Handle Telegram updates without exposing pseudonymous IDs or paths."""

    def __init__(
        self,
        *,
        client: TelegramClient,
        store: VoiceCollectionStore,
        config: BotRuntimeConfig | None = None,
    ) -> None:
        self._client = client
        self._store = store
        self._config = config or BotRuntimeConfig()

    def process_update(self, update: dict[str, Any]) -> None:
        """Process one Telegram update from polling."""

        try:
            callback = update.get("callback_query")
            if type(callback) is dict:
                self._handle_callback(cast(dict[str, Any], callback))
                return
            message = update.get("message")
            if type(message) is dict:
                update_id = _update_id(update)
                self._handle_message(cast(dict[str, Any], message), update_id=update_id)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            chat_id = _chat_id_from_update(update)
            if chat_id is not None:
                self._safe_send(chat_id, GENERIC_ERROR_TEXT)

    def _handle_callback(self, callback: dict[str, Any]) -> None:
        callback_id = callback.get("id")
        if type(callback_id) is str:
            self._client.answer_callback_query(callback_query_id=callback_id)
        message = callback.get("message")
        from_user = callback.get("from")
        data = callback.get("data")
        if (
            type(message) is not dict
            or type(from_user) is not dict
            or type(data) is not str
        ):
            return
        chat_id = _chat_id_from_message(cast(dict[str, Any], message))
        telegram_user_id = _telegram_user_id(from_user)
        if chat_id is None or telegram_user_id is None:
            return
        if data == _YES:
            session = self._store.accept(telegram_user_id)
            self._send_next_prompt(chat_id, session)
        elif data == _NO:
            self._store.decline(telegram_user_id)
            self._safe_send(chat_id, "Согласие не получено. Аудио не принимается.")

    def _handle_message(
        self, message: dict[str, Any], *, update_id: int | None
    ) -> None:
        chat_id = _chat_id_from_message(message)
        from_user = message.get("from")
        telegram_user_id = (
            _telegram_user_id(from_user) if type(from_user) is dict else None
        )
        if chat_id is None or telegram_user_id is None:
            return

        text = message.get("text")
        if type(text) is str and text.startswith("/"):
            self._handle_command(
                chat_id=chat_id, telegram_user_id=telegram_user_id, text=text
            )
            return

        session = self._store.get_session(telegram_user_id)
        if session is None or not session.accepted:
            self._safe_send(chat_id, CONSENT_REQUIRED_TEXT)
            return
        if session.is_complete:
            self._safe_send(chat_id, "Все 6 записей получены. Спасибо.")
            return
        voice = message.get("voice")
        if type(voice) is not dict:
            self._safe_send(chat_id, UNSUPPORTED_MESSAGE_TEXT)
            self._send_next_prompt(chat_id, session)
            return
        message_id = _message_id(message)
        if message_id is None:
            self._safe_send(chat_id, GENERIC_ERROR_TEXT)
            return
        self._handle_voice(
            chat_id=chat_id,
            telegram_user_id=telegram_user_id,
            voice=voice,
            message_id=message_id,
            update_id=update_id,
        )

    def _handle_command(
        self, *, chat_id: int, telegram_user_id: int, text: str
    ) -> None:
        command = text.split(maxsplit=1)[0]
        if command == "/start":
            self._safe_send(chat_id, START_TEXT, reply_markup=_consent_keyboard())
        elif command == "/status":
            session = self._store.get_session(telegram_user_id)
            if session is None or not session.accepted:
                self._safe_send(chat_id, "Согласие ещё не получено.")
            else:
                self._safe_send(
                    chat_id,
                    _progress_text(session),
                )
        elif command == "/restart":
            if self._delete_user_safely(telegram_user_id):
                self._safe_send(
                    chat_id, "Незавершённая сессия удалена. Начните заново: /start."
                )
            else:
                self._safe_send(chat_id, GENERIC_ERROR_TEXT)
        elif command == "/delete_me":
            if self._delete_user_safely(telegram_user_id):
                self._safe_send(chat_id, "Ваши локальные записи и состояние удалены.")
            else:
                self._safe_send(chat_id, GENERIC_ERROR_TEXT)
        else:
            self._safe_send(
                chat_id, "Доступные команды: /start, /status, /restart, /delete_me."
            )

    def _handle_voice(
        self,
        *,
        chat_id: int,
        telegram_user_id: int,
        voice: dict[str, Any],
        message_id: int,
        update_id: int | None,
    ) -> None:
        if self._store.has_processed_message(
            telegram_user_id=telegram_user_id,
            message_id=message_id,
        ):
            return
        if not self._voice_metadata_is_supported(voice):
            self._safe_send(chat_id, "Запись не подходит по длительности или размеру.")
            session = self._store.get_session(telegram_user_id)
            if session is not None:
                self._send_next_prompt(chat_id, session)
            return
        file_id = voice.get("file_id")
        if type(file_id) is not str or not file_id:
            self._safe_send(chat_id, GENERIC_ERROR_TEXT)
            return

        reserved = self._store.reserve_sample(telegram_user_id)
        try:
            self._client.download_file(
                file_id=file_id,
                target_path=reserved.ogg_path,
                max_bytes=self._config.max_voice_file_size_bytes,
            )
            convert_ogg_to_wav(
                source_ogg=reserved.ogg_path,
                target_wav=reserved.wav_path,
                ffmpeg_path=self._config.ffmpeg_path,
                min_seconds=self._config.min_voice_seconds,
                max_seconds=self._config.max_voice_seconds,
            )
            session, committed = self._store.commit_sample(
                telegram_user_id=telegram_user_id,
                reserved=reserved,
                message_id=message_id,
                update_id=update_id,
            )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except (AudioConversionError, TelegramApiError, VoiceCollectionStoreError):
            _cleanup_reserved_sample(reserved)
            self._safe_send(chat_id, GENERIC_ERROR_TEXT)
            return
        except Exception:
            _cleanup_reserved_sample(reserved)
            self._safe_send(chat_id, GENERIC_ERROR_TEXT)
            return

        if not committed:
            _cleanup_reserved_sample(reserved)
            return
        self._safe_send(
            chat_id,
            f"Запись принята. {_progress_text(session)}",
        )
        self._send_next_prompt(chat_id, session)

    def _voice_metadata_is_supported(self, voice: dict[str, Any]) -> bool:
        duration = voice.get("duration")
        file_size = voice.get("file_size")
        if type(duration) is not int or type(duration) is bool:
            return False
        if (
            not self._config.min_voice_seconds
            <= duration
            <= self._config.max_voice_seconds
        ):
            return False
        if file_size is not None:
            if type(file_size) is not int or type(file_size) is bool:
                return False
            if file_size <= 0 or file_size > self._config.max_voice_file_size_bytes:
                return False
        return True

    def _send_next_prompt(self, chat_id: int, session: SessionSnapshot) -> None:
        if session.is_complete:
            self._safe_send(chat_id, "Сбор завершён. Спасибо за участие.")
            return
        next_index = session.completed_samples + 1
        phrase = RESEARCH_PHRASES[session.completed_samples]
        self._safe_send(chat_id, f"Запись {next_index}/6. Произнесите: {phrase}")

    def _safe_send(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: dict[str, object] | None = None,
    ) -> None:
        try:
            self._client.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
            )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            pass

    def _delete_user_safely(self, telegram_user_id: int) -> bool:
        try:
            self._store.delete_user(telegram_user_id)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            return False
        return True


def _consent_keyboard() -> dict[str, object]:
    return {
        "inline_keyboard": [
            [
                {"text": "Согласен", "callback_data": _YES},
                {"text": "Не согласен", "callback_data": _NO},
            ]
        ]
    }


def _progress_text(session: SessionSnapshot) -> str:
    return f"Прогресс: {session.completed_samples}/{session.required_samples}."


def _chat_id_from_update(update: dict[str, Any]) -> int | None:
    message = update.get("message")
    if type(message) is dict:
        return _chat_id_from_message(cast(dict[str, Any], message))
    callback = update.get("callback_query")
    if type(callback) is dict:
        callback_message = callback.get("message")
        if type(callback_message) is dict:
            return _chat_id_from_message(cast(dict[str, Any], callback_message))
    return None


def _update_id(update: dict[str, Any]) -> int | None:
    update_id = update.get("update_id")
    if type(update_id) is int and type(update_id) is not bool:
        return update_id
    return None


def _chat_id_from_message(message: dict[str, Any]) -> int | None:
    chat = message.get("chat")
    if type(chat) is not dict:
        return None
    chat_id = chat.get("id")
    if type(chat_id) is int and type(chat_id) is not bool:
        return chat_id
    return None


def _message_id(message: dict[str, Any]) -> int | None:
    message_id = message.get("message_id")
    if type(message_id) is int and type(message_id) is not bool:
        return message_id
    return None


def _telegram_user_id(user: dict[str, Any]) -> int | None:
    user_id = user.get("id")
    if type(user_id) is int and type(user_id) is not bool:
        return user_id
    return None


def _cleanup_reserved_sample(reserved: ReservedSample) -> None:
    for path in (reserved.ogg_path, reserved.wav_path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

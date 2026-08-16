"""Small standard-library Telegram Bot API client for local polling."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast


class TelegramClient(Protocol):
    """Telegram client boundary used by the bot and tests."""

    def get_updates(
        self, *, offset: int | None, timeout_seconds: int
    ) -> list[dict[str, Any]]:
        """Return raw Telegram updates."""

    def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        reply_markup: Mapping[str, object] | None = None,
    ) -> None:
        """Send a text message."""

    def answer_callback_query(self, *, callback_query_id: str) -> None:
        """Acknowledge an inline keyboard callback."""

    def download_file(
        self,
        *,
        file_id: str,
        target_path: Path,
        max_bytes: int,
    ) -> None:
        """Download a Telegram file to a local path with a hard byte limit."""


class TelegramApiError(ValueError):
    """Stable, privacy-safe Telegram API failure."""

    def __init__(self) -> None:
        super().__init__("Telegram API request failed.")


class TelegramApiClient:
    """Minimal Telegram Bot API client using local polling and urllib."""

    def __init__(self, *, token: str, request_timeout_seconds: float = 30.0) -> None:
        if type(token) is not str or not token.strip():
            raise TelegramApiError
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._file_base_url = f"https://api.telegram.org/file/bot{token}"
        self._request_timeout_seconds = request_timeout_seconds

    def get_updates(
        self,
        *,
        offset: int | None,
        timeout_seconds: int,
    ) -> list[dict[str, Any]]:
        params: dict[str, object] = {
            "timeout": timeout_seconds,
            "allowed_updates": json.dumps(["message", "callback_query"]),
        }
        if offset is not None:
            params["offset"] = offset
        result = self._request_json("getUpdates", params)
        if type(result) is not list:
            raise TelegramApiError
        return [
            cast(dict[str, Any], update) for update in result if type(update) is dict
        ]

    def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        reply_markup: Mapping[str, object] | None = None,
    ) -> None:
        params: dict[str, object] = {"chat_id": chat_id, "text": text}
        if reply_markup is not None:
            params["reply_markup"] = json.dumps(dict(reply_markup), ensure_ascii=False)
        self._request_json("sendMessage", params)

    def answer_callback_query(self, *, callback_query_id: str) -> None:
        self._request_json(
            "answerCallbackQuery", {"callback_query_id": callback_query_id}
        )

    def download_file(
        self,
        *,
        file_id: str,
        target_path: Path,
        max_bytes: int,
    ) -> None:
        result = self._request_json("getFile", {"file_id": file_id})
        if type(result) is not dict or type(result.get("file_path")) is not str:
            raise TelegramApiError
        file_path = str(result["file_path"])
        if not file_path or file_path.startswith("/") or ".." in file_path:
            raise TelegramApiError
        failed = False
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                urllib.request.urlopen(
                    f"{self._file_base_url}/{urllib.parse.quote(file_path)}",
                    timeout=self._request_timeout_seconds,
                ) as response,
                target_path.open("wb") as output,
            ):
                downloaded = 0
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        _unlink_safely(target_path)
                        failed = True
                        break
                    output.write(cast(bytes, chunk))
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            _unlink_safely(target_path)
            failed = True
        if failed:
            raise TelegramApiError
        if not _is_telegram_voice_ogg(target_path):
            _unlink_safely(target_path)
            raise TelegramApiError

    def _request_json(self, method: str, params: Mapping[str, object]) -> object:
        data = urllib.parse.urlencode(params).encode("utf-8")
        request = urllib.request.Request(f"{self._base_url}/{method}", data=data)
        failed = False
        payload: object = None
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._request_timeout_seconds,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            failed = True
        if failed:
            raise TelegramApiError
        if type(payload) is not dict or payload.get("ok") is not True:
            raise TelegramApiError
        return payload.get("result")


def _is_telegram_voice_ogg(path: Path) -> bool:
    try:
        with path.open("rb") as file_obj:
            header = file_obj.read(4096)
    except OSError:
        return False
    return header.startswith(b"OggS") and b"OpusHead" in header


def _unlink_safely(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass

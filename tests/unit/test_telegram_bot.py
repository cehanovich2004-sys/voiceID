from __future__ import annotations

import json
import sqlite3
import wave
from pathlib import Path
from typing import Any

import pytest

from voiceid.calibration.feasibility import _load_manifest
from voiceid.telegram_bot import REQUIRED_RECORDINGS
from voiceid.telegram_bot.audio import AudioConversionError, convert_ogg_to_wav
from voiceid.telegram_bot.bot import BotRuntimeConfig, TelegramVoiceCollectionBot
from voiceid.telegram_bot.manifest import export_manifest_for_calibration
from voiceid.telegram_bot.storage import VoiceCollectionStore


class FakeTelegramClient:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str, dict[str, object] | None]] = []
        self.answered_callbacks: list[str] = []
        self.downloads = 0

    def get_updates(
        self, *, offset: int | None, timeout_seconds: int
    ) -> list[dict[str, Any]]:
        return []

    def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        reply_markup: dict[str, object] | None = None,
    ) -> None:
        self.messages.append((chat_id, text, reply_markup))

    def answer_callback_query(self, *, callback_query_id: str) -> None:
        self.answered_callbacks.append(callback_query_id)

    def download_file(self, *, file_id: str) -> bytes:
        self.downloads += 1
        return b"synthetic ogg bytes"


def test_consent_is_required_before_voice_is_downloaded(tmp_path: Path) -> None:
    client, bot = _bot(tmp_path)

    bot.process_update(_voice_update())

    assert client.downloads == 0
    assert _last_text(client) == "Сначала нужно согласиться с условиями /start."


def test_full_six_recording_flow_exports_probe_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, bot = _bot(tmp_path)
    monkeypatch.setattr(
        "voiceid.telegram_bot.bot.convert_ogg_to_wav",
        lambda *, source_ogg, target_wav, ffmpeg_path: _write_wav(target_wav),
    )

    bot.process_update(_start_update())
    bot.process_update(_consent_update())
    for _ in range(REQUIRED_RECORDINGS):
        bot.process_update(_voice_update())

    status_texts = [message[1] for message in client.messages]
    assert any("Запись 1/6" in text for text in status_texts)
    assert any("Прогресс: 6/6" in text for text in status_texts)
    assert any("Сбор завершён" in text for text in status_texts)

    store = _store(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    export_manifest_for_calibration(
        store=store,
        output_path=manifest_path,
        repository_commit_sha="a" * 40,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["samples"]) == REQUIRED_RECORDINGS
    assert {sample["partition"] for sample in manifest["samples"]} == {"CALIBRATION"}
    _load_manifest(manifest_path)


def test_unsupported_message_does_not_advance_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, bot = _bot(tmp_path)
    monkeypatch.setattr(
        "voiceid.telegram_bot.bot.convert_ogg_to_wav",
        lambda *, source_ogg, target_wav, ffmpeg_path: _write_wav(target_wav),
    )
    bot.process_update(_consent_update())

    bot.process_update(_text_update("not voice"))

    assert "Пришлите именно голосовое сообщение Telegram." in _all_text(client)
    assert _store(tmp_path).get_session(101).completed_samples == 0


def test_invalid_voice_duration_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, bot = _bot(tmp_path)
    monkeypatch.setattr(
        "voiceid.telegram_bot.bot.convert_ogg_to_wav",
        lambda *, source_ogg, target_wav, ffmpeg_path: _write_wav(target_wav),
    )
    bot.process_update(_consent_update())

    bot.process_update(_voice_update(duration=0))
    bot.process_update(_voice_update(duration=3))

    assert "не подходит по длительности или размеру" in _all_text(client)
    assert _store(tmp_path).get_session(101).completed_samples == 1


def test_ffmpeg_error_keeps_current_prompt_and_cleans_reserved_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, bot = _bot(tmp_path)

    def fail_conversion(
        *, source_ogg: Path, target_wav: Path, ffmpeg_path: str
    ) -> None:
        raise AudioConversionError

    monkeypatch.setattr("voiceid.telegram_bot.bot.convert_ogg_to_wav", fail_conversion)
    bot.process_update(_consent_update())
    bot.process_update(_voice_update())

    assert "Не удалось обработать запись" in _all_text(client)
    assert _store(tmp_path).get_session(101).completed_samples == 0
    assert not list((tmp_path / "audio").rglob("*.ogg"))
    assert not list((tmp_path / "audio").rglob("*.wav"))


def test_restart_and_delete_me_remove_local_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, bot = _bot(tmp_path)
    monkeypatch.setattr(
        "voiceid.telegram_bot.bot.convert_ogg_to_wav",
        lambda *, source_ogg, target_wav, ffmpeg_path: _write_wav(target_wav),
    )
    bot.process_update(_consent_update())
    bot.process_update(_voice_update())
    assert _store(tmp_path).get_session(101).completed_samples == 1

    bot.process_update(_command_update("/restart"))
    assert _store(tmp_path).get_session(101) is None

    bot.process_update(_consent_update())
    bot.process_update(_voice_update())
    bot.process_update(_command_update("/delete_me"))
    assert _store(tmp_path).get_session(101) is None
    assert not list((tmp_path / "audio").rglob("*"))


def test_status_does_not_disclose_identifiers_or_paths(tmp_path: Path) -> None:
    client, bot = _bot(tmp_path)
    bot.process_update(_consent_update())

    bot.process_update(_command_update("/status"))

    text = _last_text(client)
    assert "Прогресс:" in text
    assert "smp_" not in text
    assert "sub_" not in text
    assert str(tmp_path) not in text


def test_storage_recovers_after_restart(tmp_path: Path) -> None:
    client, bot = _bot(tmp_path)
    bot.process_update(_consent_update())

    restarted_client = FakeTelegramClient()
    restarted_bot = TelegramVoiceCollectionBot(
        client=restarted_client,
        store=_store(tmp_path),
        config=BotRuntimeConfig(ffmpeg_path="ffmpeg"),
    )
    restarted_bot.process_update(_command_update("/status"))

    assert "Прогресс: 0/6." == _last_text(restarted_client)


def test_convert_ogg_to_wav_uses_safe_ffmpeg_args_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class Completed:
        returncode = 0

    def fake_run(*args: object, **kwargs: object) -> Completed:
        calls.append({"args": args, "kwargs": kwargs})
        _write_wav(tmp_path / "out.wav")
        return Completed()

    monkeypatch.setattr("subprocess.run", fake_run)

    convert_ogg_to_wav(
        source_ogg=tmp_path / "in.ogg",
        target_wav=tmp_path / "out.wav",
        ffmpeg_path="ffmpeg",
        timeout_seconds=1.5,
    )

    assert calls
    command = calls[0]["args"][0]
    assert isinstance(command, list)
    assert "-ac" in command
    assert "1" in command
    assert "-ar" in command
    assert "16000" in command
    assert calls[0]["kwargs"]["timeout"] == 1.5
    assert calls[0]["kwargs"]["check"] is False
    assert calls[0]["kwargs"]["capture_output"] is True


def test_export_errors_are_generic_and_do_not_leak_paths(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError) as exc_info:
        export_manifest_for_calibration(
            store=store,
            output_path=tmp_path / "manifest.json",
            repository_commit_sha="/Users/private/TOKEN",
        )

    assert str(exc_info.value) == "Manifest export failed."
    assert "TOKEN" not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


def test_database_contains_no_manifest_file_when_incomplete(tmp_path: Path) -> None:
    _store(tmp_path).initialize()
    connection = sqlite3.connect(tmp_path / "state" / "voice_collection.sqlite3")
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    finally:
        connection.close()
    assert rows


def _bot(tmp_path: Path) -> tuple[FakeTelegramClient, TelegramVoiceCollectionBot]:
    store = _store(tmp_path)
    store.initialize()
    client = FakeTelegramClient()
    return client, TelegramVoiceCollectionBot(client=client, store=store)


def _store(tmp_path: Path) -> VoiceCollectionStore:
    return VoiceCollectionStore(
        db_path=tmp_path / "state" / "voice_collection.sqlite3",
        data_dir=tmp_path,
    )


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x01\x00" * 16000)


def _start_update() -> dict[str, Any]:
    return _command_update("/start")


def _consent_update() -> dict[str, Any]:
    return {
        "callback_query": {
            "id": "callback-1",
            "data": "consent_yes",
            "from": {"id": 101},
            "message": {"chat": {"id": 202}},
        }
    }


def _command_update(text: str) -> dict[str, Any]:
    return {
        "message": {
            "chat": {"id": 202},
            "from": {"id": 101},
            "text": text,
        }
    }


def _text_update(text: str) -> dict[str, Any]:
    return {
        "message": {
            "chat": {"id": 202},
            "from": {"id": 101},
            "text": text,
        }
    }


def _voice_update(*, duration: int = 3, file_size: int = 1024) -> dict[str, Any]:
    return {
        "message": {
            "chat": {"id": 202},
            "from": {"id": 101},
            "voice": {
                "duration": duration,
                "file_size": file_size,
                "file_id": "telegram-file-id",
            },
        }
    }


def _last_text(client: FakeTelegramClient) -> str:
    return client.messages[-1][1]


def _all_text(client: FakeTelegramClient) -> str:
    return "\n".join(message[1] for message in client.messages)

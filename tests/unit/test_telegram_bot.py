from __future__ import annotations

import json
import socket
import sqlite3
import threading
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from voiceid.calibration.feasibility import _load_manifest
from voiceid.telegram_bot import REQUIRED_RECORDINGS
from voiceid.telegram_bot.audio import AudioConversionError, convert_ogg_to_wav
from voiceid.telegram_bot.bot import BotRuntimeConfig, TelegramVoiceCollectionBot
from voiceid.telegram_bot.cli import _parse_operator_ids, _poll_once
from voiceid.telegram_bot.identification import (
    IDENTIFICATION_MIN_MARGIN,
    IDENTIFICATION_POLICY_VERSION,
    IDENTIFICATION_THRESHOLD,
    ExperimentalIdentifier,
    IdentificationResult,
    _verdict,
)
from voiceid.telegram_bot.manifest import export_manifest_for_calibration
from voiceid.telegram_bot.storage import VoiceCollectionStore
from voiceid.telegram_bot.telegram_api import TelegramApiClient, TelegramApiError


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

    def download_file(self, *, file_id: str, target_path: Path, max_bytes: int) -> None:
        self.downloads += 1
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"OggS" + b"\x00" * 24 + b"OpusHead")


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
        _fake_convert,
    )

    bot.process_update(_start_update())
    bot.process_update(_consent_update())
    for index in range(REQUIRED_RECORDINGS):
        bot.process_update(_voice_update(message_id=500 + index))

    status_texts = [message[1] for message in client.messages]
    assert any("Запись 1/6" in text for text in status_texts)
    assert any("Прогресс: 6/6" in text for text in status_texts)
    assert any("Сбор завершён" in text for text in status_texts)

    store = _store(tmp_path)
    manifest_path = _managed_manifest_path(tmp_path, "manifest")
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
        _fake_convert,
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
        _fake_convert,
    )
    bot.process_update(_consent_update())

    bot.process_update(_voice_update(duration=0, message_id=500))
    bot.process_update(_voice_update(duration=3, message_id=501))

    assert "не подходит по длительности или размеру" in _all_text(client)
    assert _store(tmp_path).get_session(101).completed_samples == 1


def test_ffmpeg_error_keeps_current_prompt_and_cleans_reserved_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, bot = _bot(tmp_path)

    def fail_conversion(
        *,
        source_ogg: Path,
        target_wav: Path,
        ffmpeg_path: str,
        min_seconds: int,
        max_seconds: int,
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
        _fake_convert,
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


def test_participant_codes_are_stable_and_not_exported(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first_subject = _complete_user(
        store,
        telegram_user_id=101,
        message_start=1000,
    )
    second_subject = _complete_user(
        store,
        telegram_user_id=202,
        message_start=2000,
    )

    assert first_subject != second_subject
    assert store.get_participant_code(101) == "P0001"
    assert store.get_participant_code(202) == "P0002"
    store.initialize()
    assert store.get_participant_code(101) == "P0001"
    store.delete_user(101)
    _complete_user(store, telegram_user_id=303, message_start=3000)
    assert store.get_participant_code(303) == "P0003"

    manifest_path = _managed_manifest_path(tmp_path, "codes")
    export_manifest_for_calibration(
        store=store,
        output_path=manifest_path,
        repository_commit_sha="a" * 40,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert all("participant_code" not in sample for sample in payload["samples"])
    _load_manifest(manifest_path)


def test_my_code_and_whoami_return_only_callers_values(tmp_path: Path) -> None:
    client, bot = _bot(tmp_path)
    bot.process_update(_consent_update())

    bot.process_update(_command_update("/my_code"))
    bot.process_update(_command_update("/whoami"))

    assert _all_text(client).count("P0001") >= 1
    assert "sub_" not in _all_text(client)
    assert "101" in _last_text(client)


def test_identify_requires_authorized_private_operator(
    tmp_path: Path,
) -> None:
    client, bot = _bot(tmp_path, operator_ids=frozenset({101}))

    bot.process_update(_command_update("/identify"))
    bot.process_update(_operator_command_update("/identify", chat_id=999))
    bot.process_update(_operator_command_update("/identify"))

    assert client.messages[0][1] == "Not authorized."
    assert client.messages[1][1] == "Not authorized."
    assert (
        client.messages[2][1]
        == "Send one voice message for experimental identification."
    )


def test_operator_allowlist_parser_fails_closed() -> None:
    assert _parse_operator_ids(None) == frozenset()
    assert _parse_operator_ids("") == frozenset()
    assert _parse_operator_ids("101,202") == frozenset({101, 202})
    assert _parse_operator_ids("101,abc") == frozenset()
    assert _parse_operator_ids("0") == frozenset()
    assert _parse_operator_ids("-1") == frozenset()


def test_identify_returns_safe_outcome_and_does_not_create_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    _complete_user(store, telegram_user_id=201, message_start=2000)
    _complete_user(store, telegram_user_id=202, message_start=3000)
    identifier = _FakeIdentifier("IDENTIFIED: P0001")
    client = FakeTelegramClient()
    bot = TelegramVoiceCollectionBot(
        client=client,
        store=store,
        operator_ids=frozenset({101}),
        identifier=identifier,
    )
    monkeypatch.setattr("voiceid.telegram_bot.bot.convert_ogg_to_wav", _fake_convert)

    bot.process_update(_operator_command_update("/identify"))
    bot.process_update(_operator_voice_update(message_id=900, update_id=901))

    assert _last_text(client) == "IDENTIFIED: P0001"
    assert identifier.calls == 1
    assert not list((tmp_path / "tmp").rglob("*"))
    assert len(store.export_rows()) == REQUIRED_RECORDINGS * 2


def test_identify_duplicate_query_is_noop_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    _complete_user(store, telegram_user_id=201, message_start=2000)
    _complete_user(store, telegram_user_id=202, message_start=3000)
    identifier = _FakeIdentifier("UNKNOWN")
    client = FakeTelegramClient()
    bot = TelegramVoiceCollectionBot(
        client=client,
        store=store,
        operator_ids=frozenset({101}),
        identifier=identifier,
    )
    monkeypatch.setattr("voiceid.telegram_bot.bot.convert_ogg_to_wav", _fake_convert)

    bot.process_update(_operator_command_update("/identify"))
    update = _operator_voice_update(message_id=900, update_id=901)
    bot.process_update(update)
    bot.process_update(_operator_command_update("/identify"))
    bot.process_update(update)

    assert identifier.calls == 1
    assert [message[1] for message in client.messages].count("UNKNOWN") == 1


def test_identify_unavailable_with_one_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    _complete_user(store, telegram_user_id=201, message_start=2000)
    client = FakeTelegramClient()
    bot = TelegramVoiceCollectionBot(
        client=client,
        store=store,
        operator_ids=frozenset({101}),
        identifier=_FakeIdentifier("IDENTIFICATION UNAVAILABLE"),
    )
    monkeypatch.setattr("voiceid.telegram_bot.bot.convert_ogg_to_wav", _fake_convert)

    bot.process_update(_operator_command_update("/identify"))
    bot.process_update(_operator_voice_update(message_id=900, update_id=901))

    assert _last_text(client) == "IDENTIFICATION UNAVAILABLE"


def test_identification_policy_boundaries_are_versioned() -> None:
    assert IDENTIFICATION_POLICY_VERSION == "telegram-identification-policy-v1"
    assert IDENTIFICATION_THRESHOLD == 0.4
    assert IDENTIFICATION_MIN_MARGIN == 0.05
    assert (
        _verdict(top_score=0.4, second_score=0.35, participant_code="P0001")
        == "IDENTIFIED: P0001"
    )
    assert (
        _verdict(top_score=0.39, second_score=0.1, participant_code="P0001")
        == "UNKNOWN"
    )
    assert (
        _verdict(top_score=0.6, second_score=0.551, participant_code="P0001")
        == "AMBIGUOUS"
    )


def test_identification_does_not_patch_global_socket(
    tmp_path: Path,
) -> None:
    create_connection = socket.create_connection
    socket_connect = socket.socket.connect
    identifier = ExperimentalIdentifier(model_cache_dir=tmp_path)

    result = identifier.identify(query_wav_path=tmp_path / "missing.wav", profiles=())

    assert result.text == "IDENTIFICATION UNAVAILABLE"
    assert socket.create_connection is create_connection
    assert socket.socket.connect is socket_connect


def test_poll_once_advances_offset_when_update_processing_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class OneUpdateClient:
        def get_updates(
            self, *, offset: int | None, timeout_seconds: int
        ) -> list[dict[str, Any]]:
            return [{"update_id": 777, "message": {"text": "/identify"}}]

    class FailingBot:
        def process_update(self, update: dict[str, Any]) -> None:
            raise RuntimeError("TOKEN_CANARY_PATH_PAYLOAD")

    caplog.set_level("WARNING", logger="voiceid.telegram_bot")

    offset = _poll_once(
        client=OneUpdateClient(),
        bot=FailingBot(),
        offset=10,
        timeout_seconds=0,
    )

    assert offset == 778
    assert "telegram_polling_error" in caplog.text
    assert "TOKEN_CANARY" not in caplog.text


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

    assert "Прогресс: 0/6. Код: P0001." == _last_text(restarted_client)


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


def test_convert_ogg_rejects_short_and_long_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        returncode = 0

    def short_run(*args: object, **kwargs: object) -> Completed:
        _write_wav(tmp_path / "short.wav", frames=100)
        return Completed()

    monkeypatch.setattr("subprocess.run", short_run)
    with pytest.raises(AudioConversionError):
        convert_ogg_to_wav(
            source_ogg=tmp_path / "in.ogg",
            target_wav=tmp_path / "short.wav",
        )
    assert not (tmp_path / "short.wav").exists()

    def long_run(*args: object, **kwargs: object) -> Completed:
        _write_wav(tmp_path / "long.wav", frames=16000 * 61)
        return Completed()

    monkeypatch.setattr("subprocess.run", long_run)
    with pytest.raises(AudioConversionError):
        convert_ogg_to_wav(
            source_ogg=tmp_path / "in.ogg",
            target_wav=tmp_path / "long.wav",
        )
    assert not (tmp_path / "long.wav").exists()


def test_telegram_download_streams_limit_and_validates_ogg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TelegramApiClient(token="TOKEN_CANARY")
    responses = [
        _JsonResponse({"ok": True, "result": {"file_path": "voice/file.ogg"}}),
        _BytesResponse([b"OggS", b"\x00" * 20, b"OpusHead"]),
    ]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: responses.pop(0),
    )

    target = tmp_path / "voice.ogg"
    client.download_file(file_id="file-id", target_path=target, max_bytes=64)

    assert target.read_bytes().startswith(b"OggS")


def test_telegram_download_rejects_fake_mime_and_truncated_ogg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TelegramApiClient(token="TOKEN_CANARY")
    responses = [
        _JsonResponse({"ok": True, "result": {"file_path": "voice/file.ogg"}}),
        _BytesResponse([b"text/plain payload"]),
    ]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: responses.pop(0),
    )

    target = tmp_path / "voice.ogg"
    with pytest.raises(TelegramApiError) as exc_info:
        client.download_file(file_id="file-id", target_path=target, max_bytes=64)

    assert str(exc_info.value) == "Telegram API request failed."
    assert "TOKEN_CANARY" not in str(exc_info.value)
    assert not target.exists()


def test_telegram_download_deletes_partial_file_when_chunk_crosses_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TelegramApiClient(token="TOKEN_CANARY")
    responses = [
        _JsonResponse({"ok": True, "result": {"file_path": "voice/file.ogg"}}),
        _BytesResponse([b"OggS", b"x" * 65]),
    ]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: responses.pop(0),
    )

    target = tmp_path / "voice.ogg"
    with pytest.raises(TelegramApiError):
        client.download_file(file_id="file-id", target_path=target, max_bytes=64)

    assert not target.exists()


def test_export_errors_are_generic_and_do_not_leak_paths(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError) as exc_info:
        export_manifest_for_calibration(
            store=store,
            output_path=_managed_manifest_path(tmp_path, "manifest"),
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


def test_delete_me_removes_completed_samples_and_manifest_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, bot = _bot(tmp_path)
    monkeypatch.setattr(
        "voiceid.telegram_bot.bot.convert_ogg_to_wav",
        _fake_convert,
    )
    bot.process_update(_consent_update())
    for index in range(REQUIRED_RECORDINGS):
        bot.process_update(_voice_update(message_id=1000 + index))

    first_manifest = _managed_manifest_path(tmp_path, "before")
    export_manifest_for_calibration(
        store=_store(tmp_path),
        output_path=first_manifest,
        repository_commit_sha="a" * 40,
    )
    assert len(json.loads(first_manifest.read_text(encoding="utf-8"))["samples"]) == 6

    bot.process_update(_command_update("/delete_me"))
    second_manifest = _managed_manifest_path(tmp_path, "after")
    export_manifest_for_calibration(
        store=_store(tmp_path),
        output_path=second_manifest,
        repository_commit_sha="a" * 40,
    )
    assert json.loads(second_manifest.read_text(encoding="utf-8"))["samples"] == []
    assert not list((tmp_path / "audio").rglob("*.ogg"))
    assert not list((tmp_path / "audio").rglob("*.wav"))
    assert _store(tmp_path).get_session(101) is None


def test_delete_me_rewrites_managed_manifest_and_keeps_other_participants(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    deleted_subject = _complete_user(store, telegram_user_id=101, message_start=1000)
    kept_subject = _complete_user(store, telegram_user_id=202, message_start=2000)
    manifest_path = _managed_manifest_path(tmp_path, "managed")
    export_manifest_for_calibration(
        store=store,
        output_path=manifest_path,
        repository_commit_sha="a" * 40,
    )
    assert len(json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]) == 12

    client = FakeTelegramClient()
    bot = TelegramVoiceCollectionBot(client=client, store=store)
    bot.process_update(_command_update("/delete_me"))

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload["samples"]) == 6
    assert {sample["subject_id"] for sample in payload["samples"]} == {kept_subject}
    assert deleted_subject not in json.dumps(payload)
    _load_manifest(manifest_path)


def test_delete_me_manifest_write_failure_is_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    _complete_user(store, telegram_user_id=101, message_start=1000)
    manifest_path = _managed_manifest_path(tmp_path, "managed")
    export_manifest_for_calibration(
        store=store,
        output_path=manifest_path,
        repository_commit_sha="a" * 40,
    )
    original_bytes = manifest_path.read_bytes()
    original_dump = json.dump
    calls = 0

    def flaky_dump(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("TOKEN_MANIFEST_WRITE_CANARY")
        original_dump(*args, **kwargs)

    monkeypatch.setattr(json, "dump", flaky_dump)
    client = FakeTelegramClient()
    bot = TelegramVoiceCollectionBot(client=client, store=store)
    bot.process_update(_command_update("/delete_me"))

    assert "Не удалось обработать запись" in _last_text(client)
    assert manifest_path.read_bytes() == original_bytes
    assert _store(tmp_path).get_session(101) is not None
    assert not list((tmp_path / "manifests").glob("*.tmp"))

    monkeypatch.setattr(json, "dump", original_dump)
    bot.process_update(_command_update("/delete_me"))
    assert "удалены" in _last_text(client)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["samples"] == []
    assert _store(tmp_path).get_session(101) is None


def test_delete_me_manifest_replace_failure_is_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    _complete_user(store, telegram_user_id=101, message_start=1000)
    manifest_path = _managed_manifest_path(tmp_path, "managed")
    export_manifest_for_calibration(
        store=store,
        output_path=manifest_path,
        repository_commit_sha="a" * 40,
    )
    original_bytes = manifest_path.read_bytes()
    original_replace = Path.replace
    calls = 0

    def flaky_replace(self: Path, target: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("TOKEN_MANIFEST_REPLACE_CANARY")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    client = FakeTelegramClient()
    bot = TelegramVoiceCollectionBot(client=client, store=store)
    bot.process_update(_command_update("/delete_me"))

    assert "Не удалось обработать запись" in _last_text(client)
    assert manifest_path.read_bytes() == original_bytes
    assert _store(tmp_path).get_session(101) is not None
    assert not list((tmp_path / "manifests").glob("*.tmp"))

    monkeypatch.setattr(Path, "replace", original_replace)
    bot.process_update(_command_update("/delete_me"))
    assert "удалены" in _last_text(client)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["samples"] == []


def test_delete_me_rewrites_multiple_managed_manifests(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize()
    _complete_user(store, telegram_user_id=101, message_start=1000)
    manifests = [
        _managed_manifest_path(tmp_path, "first"),
        _managed_manifest_path(tmp_path, "second"),
    ]
    for manifest_path in manifests:
        export_manifest_for_calibration(
            store=store,
            output_path=manifest_path,
            repository_commit_sha="a" * 40,
        )

    TelegramVoiceCollectionBot(
        client=FakeTelegramClient(),
        store=store,
    ).process_update(_command_update("/delete_me"))

    for manifest_path in manifests:
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["samples"] == []


def test_export_rejects_path_traversal_and_symlink_escape(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize()
    _complete_user(store, telegram_user_id=101, message_start=1000)

    with pytest.raises(ValueError) as traversal:
        export_manifest_for_calibration(
            store=store,
            output_path=tmp_path / "manifests" / ".." / "evil.manifest.json",
            repository_commit_sha="a" * 40,
        )
    assert str(traversal.value) == "Manifest export failed."

    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = tmp_path / "manifests" / "escape.manifest.json"
    (tmp_path / "manifests").mkdir(exist_ok=True)
    symlink.symlink_to(outside / "escape.manifest.json")
    with pytest.raises(ValueError) as symlink_error:
        export_manifest_for_calibration(
            store=store,
            output_path=symlink,
            repository_commit_sha="a" * 40,
        )
    assert str(symlink_error.value) == "Manifest export failed."


def test_delete_me_is_recoverable_after_file_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, bot = _bot(tmp_path)
    monkeypatch.setattr(
        "voiceid.telegram_bot.bot.convert_ogg_to_wav",
        _fake_convert,
    )
    bot.process_update(_consent_update())
    bot.process_update(_voice_update())
    original_unlink = Path.unlink
    calls = 0

    def flaky_unlink(self: Path, *args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("TOKEN_DELETE_CANARY")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    bot.process_update(_command_update("/delete_me"))
    assert "Не удалось обработать запись" in _last_text(client)
    assert _store(tmp_path).get_session(101) is not None

    monkeypatch.setattr(Path, "unlink", original_unlink)
    bot.process_update(_command_update("/delete_me"))
    assert "удалены" in _last_text(client)
    assert _store(tmp_path).get_session(101) is None


def test_delete_me_is_idempotent_for_missing_user(tmp_path: Path) -> None:
    client, bot = _bot(tmp_path)

    bot.process_update(_command_update("/delete_me"))
    bot.process_update(_command_update("/delete_me"))

    assert "удалены" in _last_text(client)


def test_duplicate_update_and_message_are_noop_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "voiceid.telegram_bot.bot.convert_ogg_to_wav",
        _fake_convert,
    )
    client, bot = _bot(tmp_path)
    bot.process_update(_consent_update())
    update = _voice_update(update_id=900, message_id=901)
    bot.process_update(update)

    restarted = TelegramVoiceCollectionBot(client=client, store=_store(tmp_path))
    restarted.process_update(update)

    assert _store(tmp_path).get_session(101).completed_samples == 1


def test_concurrent_duplicate_prompt_commits_only_one_sample(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize()
    store.accept(101)
    first = store.reserve_sample(101)
    second = store.reserve_sample(101)
    _write_wav(first.wav_path)
    first.ogg_path.parent.mkdir(parents=True, exist_ok=True)
    first.ogg_path.write_bytes(b"OggS OpusHead")
    _write_wav(second.wav_path)
    second.ogg_path.parent.mkdir(parents=True, exist_ok=True)
    second.ogg_path.write_bytes(b"OggS OpusHead")
    results: list[bool] = []

    def commit(reserved: object, message_id: int) -> None:
        _, committed = store.commit_sample(
            telegram_user_id=101,
            reserved=reserved,  # type: ignore[arg-type]
            message_id=message_id,
            update_id=message_id,
        )
        results.append(committed)

    threads = [
        threading.Thread(target=commit, args=(first, 1)),
        threading.Thread(target=commit, args=(second, 2)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [False, True]
    assert store.get_session(101).completed_samples == 1


def test_cli_polling_boundary_logs_generic_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingClient:
        def get_updates(
            self,
            *,
            offset: int | None,
            timeout_seconds: int,
        ) -> list[dict[str, Any]]:
            raise RuntimeError("/Users/private/TOKEN_POLLING_CANARY")

    caplog.set_level("WARNING", logger="voiceid.telegram_bot")
    _, bot = _bot(tmp_path)
    offset = _poll_once(
        client=FailingClient(),
        bot=bot,
        offset=100,
        timeout_seconds=0,
    )

    assert offset == 100
    assert "telegram_polling_error" in caplog.text
    assert "TOKEN_POLLING_CANARY" not in caplog.text


def test_network_exception_does_not_keep_sensitive_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TelegramApiClient(token="TOKEN_NET_CANARY")

    def fail_urlopen(*args: object, **kwargs: object) -> object:
        raise RuntimeError("https://api.telegram.org/botTOKEN_NET_CANARY/path")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    with pytest.raises(TelegramApiError) as exc_info:
        client.download_file(
            file_id="file-id",
            target_path=Path("/tmp/not-written.ogg"),
            max_bytes=100,
        )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "TOKEN_NET_CANARY" not in str(exc_info.value)


def _bot(
    tmp_path: Path,
    *,
    operator_ids: frozenset[int] = frozenset(),
) -> tuple[FakeTelegramClient, TelegramVoiceCollectionBot]:
    store = _store(tmp_path)
    store.initialize()
    client = FakeTelegramClient()
    return client, TelegramVoiceCollectionBot(
        client=client,
        store=store,
        operator_ids=operator_ids,
    )


def _store(tmp_path: Path) -> VoiceCollectionStore:
    return VoiceCollectionStore(
        db_path=tmp_path / "state" / "voice_collection.sqlite3",
        data_dir=tmp_path,
    )


def _managed_manifest_path(tmp_path: Path, name: str) -> Path:
    return tmp_path / "manifests" / f"{name}.manifest.json"


def _complete_user(
    store: VoiceCollectionStore,
    *,
    telegram_user_id: int,
    message_start: int,
) -> str:
    store.initialize()
    session = store.accept(telegram_user_id)
    for offset in range(REQUIRED_RECORDINGS):
        reserved = store.reserve_sample(telegram_user_id)
        reserved.ogg_path.parent.mkdir(parents=True, exist_ok=True)
        reserved.ogg_path.write_bytes(b"OggS" + b"\x00" * 24 + b"OpusHead")
        _write_wav(reserved.wav_path)
        store.commit_sample(
            telegram_user_id=telegram_user_id,
            reserved=reserved,
            message_id=message_start + offset,
            update_id=message_start + offset,
        )
    return session.subject_id


def _write_wav(path: Path, *, frames: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x01\x00" * frames)


def _fake_convert(
    *,
    source_ogg: Path,
    target_wav: Path,
    ffmpeg_path: str,
    min_seconds: int,
    max_seconds: int,
) -> None:
    _write_wav(target_wav)


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


def _operator_command_update(text: str, *, chat_id: int = 101) -> dict[str, Any]:
    return {
        "message": {
            "chat": {"id": chat_id},
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


def _voice_update(
    *,
    duration: int = 3,
    file_size: int = 1024,
    update_id: int = 301,
    message_id: int = 401,
) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "chat": {"id": 202},
            "from": {"id": 101},
            "voice": {
                "duration": duration,
                "file_size": file_size,
                "file_id": "telegram-file-id",
            },
        },
    }


def _operator_voice_update(
    *,
    duration: int = 3,
    file_size: int = 1024,
    update_id: int = 901,
    message_id: int = 900,
) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "chat": {"id": 101},
            "from": {"id": 101},
            "voice": {
                "duration": duration,
                "file_size": file_size,
                "file_id": "operator-file-id",
            },
        },
    }


def _last_text(client: FakeTelegramClient) -> str:
    return client.messages[-1][1]


def _all_text(client: FakeTelegramClient) -> str:
    return "\n".join(message[1] for message in client.messages)


class _JsonResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def __enter__(self) -> _JsonResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _BytesResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._stream = BytesIO(b"".join(chunks))

    def __enter__(self) -> _BytesResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


class _FakeIdentifier:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def identify(self, **kwargs: object) -> IdentificationResult:
        self.calls += 1
        return IdentificationResult(self._text)

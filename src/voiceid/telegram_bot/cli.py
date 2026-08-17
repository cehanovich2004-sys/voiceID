"""Command-line entry points for the local Telegram collection MVP."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Final

from voiceid.telegram_bot.bot import TelegramVoiceCollectionBot
from voiceid.telegram_bot.config import DEFAULT_DATA_DIR, DEFAULT_MODEL_CACHE_DIR
from voiceid.telegram_bot.identification import ExperimentalIdentifier
from voiceid.telegram_bot.manifest import export_manifest_for_calibration
from voiceid.telegram_bot.storage import VoiceCollectionStore
from voiceid.telegram_bot.telegram_api import TelegramApiClient, TelegramClient

_LOGGER = logging.getLogger("voiceid.telegram_bot")
_OPERATOR_ENV: Final = "VOICEID_TELEGRAM_OPERATOR_IDS"


def main(argv: list[str] | None = None) -> int:
    """Run local Telegram polling or export a feasibility manifest."""

    parser = argparse.ArgumentParser(
        description="Local VoiceID Telegram voice collection MVP."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run local Telegram polling.")
    run_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    run_parser.add_argument("--model-cache-dir", default=str(DEFAULT_MODEL_CACHE_DIR))
    run_parser.add_argument("--poll-timeout", type=int, default=25)

    export_parser = subparsers.add_parser(
        "export-manifest",
        help="Export a local feasibility-probe manifest.",
    )
    export_parser.add_argument("output")
    export_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    export_parser.add_argument("--repository-commit-sha", required=True)

    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    store = VoiceCollectionStore(
        db_path=data_dir / "state" / "voice_collection.sqlite3",
        data_dir=data_dir,
    )
    store.initialize()

    if args.command == "export-manifest":
        export_manifest_for_calibration(
            store=store,
            output_path=Path(args.output),
            repository_commit_sha=args.repository_commit_sha,
        )
        print("Manifest exported.")
        return 0

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if token is None or not token.strip():
        print("TELEGRAM_BOT_TOKEN is required.")
        return 2
    client = TelegramApiClient(token=token)
    bot = TelegramVoiceCollectionBot(
        client=client,
        store=store,
        operator_ids=_parse_operator_ids(os.environ.get(_OPERATOR_ENV)),
        identifier=ExperimentalIdentifier(model_cache_dir=Path(args.model_cache_dir)),
    )
    offset: int | None = None
    while True:
        offset = _poll_once(
            client=client,
            bot=bot,
            offset=offset,
            timeout_seconds=args.poll_timeout,
        )


def _poll_once(
    *,
    client: TelegramClient,
    bot: TelegramVoiceCollectionBot,
    offset: int | None,
    timeout_seconds: int,
) -> int | None:
    try:
        updates = client.get_updates(
            offset=offset,
            timeout_seconds=timeout_seconds,
        )
        next_offset = offset
        for update in updates:
            update_id = update.get("update_id")
            if type(update_id) is int and type(update_id) is not bool:
                next_offset = update_id + 1
            bot.process_update(update)
        return next_offset
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        _LOGGER.warning("telegram_polling_error")
        return offset


def _parse_operator_ids(raw: str | None) -> frozenset[int]:
    if raw is None or not raw.strip():
        return frozenset()
    values: set[int] = set()
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text or not text.isdecimal():
            return frozenset()
        value = int(text)
        if value <= 0:
            return frozenset()
        values.add(value)
    return frozenset(values)


if __name__ == "__main__":
    raise SystemExit(main())

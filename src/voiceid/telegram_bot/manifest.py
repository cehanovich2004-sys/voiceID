"""Manifest export for Telegram-collected local VoiceID samples."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

from voiceid.calibration import CalibrationPartition
from voiceid.telegram_bot.storage import VoiceCollectionStore, VoiceCollectionStoreError

_SHA_RE: Final = re.compile(r"^[0-9a-fA-F]{40}$")
_MANIFEST_ERROR: Final = "Manifest export failed."


class ManifestExportError(ValueError):
    """Stable, privacy-safe manifest export failure."""

    def __init__(self) -> None:
        super().__init__(_MANIFEST_ERROR)


def export_manifest_for_calibration(
    *,
    store: VoiceCollectionStore,
    output_path: Path,
    repository_commit_sha: str,
) -> None:
    """Write a feasibility-probe-compatible local manifest for completed sessions."""

    if (
        type(repository_commit_sha) is not str
        or _SHA_RE.fullmatch(repository_commit_sha) is None
    ):
        raise ManifestExportError
    try:
        rows = store.export_rows()
        samples = [
            {
                "sample_id": str(row["sample_id"]),
                "subject_id": str(row["subject_id"]),
                "source_group_id": str(row["source_group_id"]),
                "partition": CalibrationPartition.CALIBRATION.value,
                "wav_path": str(row["wav_path"]),
            }
            for row in rows
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "repository_commit_sha": repository_commit_sha,
                    "samples": samples,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except (ManifestExportError, VoiceCollectionStoreError):
        raise ManifestExportError from None
    except Exception:
        raise ManifestExportError from None

"""SQLite state for the local Telegram voice collection MVP."""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from voiceid.calibration import CalibrationPartition
from voiceid.telegram_bot.config import REQUIRED_RECORDINGS
from voiceid.telegram_bot.ids import new_sample_id, new_source_group_id, new_subject_id


class VoiceCollectionStoreError(ValueError):
    """Stable, privacy-safe storage failure."""

    def __init__(self) -> None:
        super().__init__("Voice collection storage failed.")


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Privacy-safe progress snapshot for one Telegram participant."""

    subject_id: str
    accepted: bool
    completed_samples: int
    required_samples: int

    @property
    def is_complete(self) -> bool:
        """Return whether the participant has completed the required recordings."""

        return self.completed_samples >= self.required_samples

    def __repr__(self) -> str:
        return (
            "SessionSnapshot("
            f"accepted={self.accepted}, "
            f"completed_samples={self.completed_samples}, "
            f"required_samples={self.required_samples})"
        )


@dataclass(frozen=True, slots=True)
class ReservedSample:
    """Internal local sample reservation with filesystem paths."""

    sample_id: str
    source_group_id: str
    ogg_path: Path
    wav_path: Path

    def __repr__(self) -> str:
        return "ReservedSample(redacted=True)"


class VoiceCollectionStore:
    """Persist Telegram collection state without exposing paths in public replies."""

    def __init__(self, *, db_path: Path, data_dir: Path) -> None:
        self._db_path = db_path
        self._data_dir = data_dir

    @property
    def db_path(self) -> Path:
        """Return the local SQLite path for operator commands."""

        return self._db_path

    @property
    def data_dir(self) -> Path:
        """Return the local data root for operator commands."""

        return self._data_dir

    def initialize(self) -> None:
        """Create local SQLite tables and storage directories."""

        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS telegram_participants (
                        telegram_user_id INTEGER PRIMARY KEY,
                        subject_id TEXT NOT NULL UNIQUE,
                        accepted INTEGER NOT NULL,
                        required_samples INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS voice_samples (
                        telegram_user_id INTEGER NOT NULL,
                        sample_id TEXT PRIMARY KEY,
                        subject_id TEXT NOT NULL,
                        source_group_id TEXT NOT NULL,
                        partition TEXT NOT NULL,
                        prompt_index INTEGER NOT NULL,
                        ogg_path TEXT NOT NULL,
                        wav_path TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(telegram_user_id)
                            REFERENCES telegram_participants(telegram_user_id)
                            ON DELETE CASCADE
                    );
                    """
                )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            raise VoiceCollectionStoreError from exc

    def get_session(self, telegram_user_id: int) -> SessionSnapshot | None:
        """Return a privacy-safe progress snapshot, if a session exists."""

        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT subject_id, accepted, required_samples
                    FROM telegram_participants
                    WHERE telegram_user_id = ?
                    """,
                    (telegram_user_id,),
                ).fetchone()
                if row is None:
                    return None
                count_row = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM voice_samples
                    WHERE telegram_user_id = ?
                    """,
                    (telegram_user_id,),
                ).fetchone()
            return SessionSnapshot(
                subject_id=str(row["subject_id"]),
                accepted=bool(row["accepted"]),
                completed_samples=int(count_row[0]),
                required_samples=int(row["required_samples"]),
            )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            raise VoiceCollectionStoreError from exc

    def accept(self, telegram_user_id: int) -> SessionSnapshot:
        """Create a consented pseudonymous session if needed."""

        try:
            current = self.get_session(telegram_user_id)
            if current is not None and current.accepted:
                return current
            subject_id = new_subject_id()
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO telegram_participants (
                        telegram_user_id, subject_id, accepted, required_samples
                    )
                    VALUES (?, ?, 1, ?)
                    """,
                    (telegram_user_id, subject_id, REQUIRED_RECORDINGS),
                )
            return self.get_session(telegram_user_id) or SessionSnapshot(
                subject_id=subject_id,
                accepted=True,
                completed_samples=0,
                required_samples=REQUIRED_RECORDINGS,
            )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            raise VoiceCollectionStoreError from exc

    def decline(self, telegram_user_id: int) -> None:
        """Remove any unfinished local session after consent refusal."""

        self.delete_user(telegram_user_id)

    def reserve_sample(self, telegram_user_id: int) -> ReservedSample:
        """Reserve pseudonymous local paths for the next accepted voice message."""

        session = self.get_session(telegram_user_id)
        if session is None or not session.accepted or session.is_complete:
            raise VoiceCollectionStoreError
        sample_id = new_sample_id()
        source_group_id = new_source_group_id()
        sample_dir = self._data_dir / "audio" / session.subject_id / sample_id
        return ReservedSample(
            sample_id=sample_id,
            source_group_id=source_group_id,
            ogg_path=sample_dir / "voice.ogg",
            wav_path=sample_dir / "voice.wav",
        )

    def commit_sample(
        self,
        *,
        telegram_user_id: int,
        reserved: ReservedSample,
    ) -> SessionSnapshot:
        """Persist a converted sample after the WAV has been verified."""

        session = self.get_session(telegram_user_id)
        if session is None or not session.accepted or session.is_complete:
            raise VoiceCollectionStoreError
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO voice_samples (
                        telegram_user_id, sample_id, subject_id, source_group_id,
                        partition, prompt_index, ogg_path, wav_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        telegram_user_id,
                        reserved.sample_id,
                        session.subject_id,
                        reserved.source_group_id,
                        CalibrationPartition.CALIBRATION.value,
                        session.completed_samples + 1,
                        str(reserved.ogg_path),
                        str(reserved.wav_path),
                    ),
                )
            return self.get_session(telegram_user_id) or session
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            raise VoiceCollectionStoreError from exc

    def delete_user(self, telegram_user_id: int) -> None:
        """Delete local records and audio for one Telegram user."""

        try:
            subject_ids = self._subject_ids_for_user(telegram_user_id)
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM voice_samples WHERE telegram_user_id = ?",
                    (telegram_user_id,),
                )
                connection.execute(
                    "DELETE FROM telegram_participants WHERE telegram_user_id = ?",
                    (telegram_user_id,),
                )
            for subject_id in subject_ids:
                shutil.rmtree(
                    self._data_dir / "audio" / subject_id,
                    ignore_errors=True,
                )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            raise VoiceCollectionStoreError from exc

    def export_rows(self) -> tuple[sqlite3.Row, ...]:
        """Return completed local sample rows for manifest export."""

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT sample_id, subject_id, source_group_id, partition, wav_path
                    FROM voice_samples
                    WHERE telegram_user_id IN (
                        SELECT telegram_user_id
                        FROM voice_samples
                        GROUP BY telegram_user_id
                        HAVING COUNT(*) >= ?
                    )
                    ORDER BY subject_id, prompt_index
                    """,
                    (REQUIRED_RECORDINGS,),
                ).fetchall()
            return tuple(rows)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            raise VoiceCollectionStoreError from exc

    def _subject_ids_for_user(self, telegram_user_id: int) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT subject_id FROM telegram_participants WHERE telegram_user_id = ?
                UNION
                SELECT subject_id FROM voice_samples WHERE telegram_user_id = ?
                """,
                (telegram_user_id, telegram_user_id),
            ).fetchall()
        return tuple(str(row["subject_id"]) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

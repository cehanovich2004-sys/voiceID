"""SQLite state for the local Telegram voice collection MVP."""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from voiceid.calibration import CalibrationPartition, CalibrationSampleRecord
from voiceid.telegram_bot.config import REQUIRED_RECORDINGS
from voiceid.telegram_bot.ids import new_sample_id, new_source_group_id, new_subject_id

_PARTICIPANT_CODE_RE: Final = re.compile(r"^P[0-9]{4}$")
_MAX_PARTICIPANT_CODE: Final = 9999


class VoiceCollectionStoreError(ValueError):
    """Stable, privacy-safe storage failure."""

    def __init__(self) -> None:
        super().__init__("Voice collection storage failed.")


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Privacy-safe progress snapshot for one Telegram participant."""

    subject_id: str
    participant_code: str | None
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
    prompt_index: int
    ogg_path: Path
    wav_path: Path

    def __repr__(self) -> str:
        return "ReservedSample(redacted=True)"


@dataclass(frozen=True, slots=True)
class EnrollmentSample:
    """Internal completed enrollment sample path for operator identification."""

    prompt_index: int
    wav_path: Path

    def __repr__(self) -> str:
        return "EnrollmentSample(redacted=True)"


@dataclass(frozen=True, slots=True)
class EnrollmentProfile:
    """Internal completed profile candidate for exploratory identification."""

    participant_code: str
    samples: tuple[EnrollmentSample, ...]

    def __repr__(self) -> str:
        return "EnrollmentProfile(redacted=True)"


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
                        UNIQUE(telegram_user_id, prompt_index),
                        UNIQUE(telegram_user_id, sample_id),
                        UNIQUE(telegram_user_id, source_group_id)
                    );
                    CREATE TABLE IF NOT EXISTS processed_voice_messages (
                        telegram_user_id INTEGER NOT NULL,
                        message_id INTEGER NOT NULL,
                        update_id INTEGER,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY(telegram_user_id, message_id),
                        FOREIGN KEY(telegram_user_id)
                            REFERENCES telegram_participants(telegram_user_id)
                            ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS managed_manifest_artifacts (
                        manifest_path TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS participant_code_reservations (
                        participant_code TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS participant_codes (
                        telegram_user_id INTEGER PRIMARY KEY,
                        subject_id TEXT NOT NULL UNIQUE,
                        participant_code TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(telegram_user_id)
                            REFERENCES telegram_participants(telegram_user_id)
                            ON DELETE CASCADE,
                        FOREIGN KEY(participant_code)
                            REFERENCES participant_code_reservations(participant_code)
                    );
                    CREATE TABLE IF NOT EXISTS processed_identification_messages (
                        telegram_user_id INTEGER NOT NULL,
                        message_id INTEGER NOT NULL,
                        update_id INTEGER,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY(telegram_user_id, message_id)
                    );
                    """
                )
                self._migrate_participant_codes(connection)
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
                    SELECT p.subject_id, c.participant_code, p.accepted,
                           p.required_samples
                    FROM telegram_participants p
                    LEFT JOIN participant_codes c
                        ON c.telegram_user_id = p.telegram_user_id
                    WHERE p.telegram_user_id = ?
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
                participant_code=(
                    str(row["participant_code"])
                    if row["participant_code"] is not None
                    else None
                ),
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
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT OR REPLACE INTO telegram_participants (
                        telegram_user_id, subject_id, accepted, required_samples
                    )
                    VALUES (?, ?, 1, ?)
                    """,
                    (telegram_user_id, subject_id, REQUIRED_RECORDINGS),
                )
                self._assign_participant_code(
                    connection,
                    telegram_user_id=telegram_user_id,
                    subject_id=subject_id,
                )
            return self.get_session(telegram_user_id) or SessionSnapshot(
                subject_id=subject_id,
                participant_code=None,
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
            prompt_index=session.completed_samples + 1,
            ogg_path=sample_dir / "voice.ogg",
            wav_path=sample_dir / "voice.wav",
        )

    def get_participant_code(self, telegram_user_id: int) -> str | None:
        """Return the caller's participant code, if one exists."""

        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT participant_code
                    FROM participant_codes
                    WHERE telegram_user_id = ?
                    """,
                    (telegram_user_id,),
                ).fetchone()
            if row is None:
                return None
            code = str(row["participant_code"])
            if not _participant_code_is_valid(code):
                raise VoiceCollectionStoreError
            return code
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            raise VoiceCollectionStoreError from exc

    def completed_enrollment_profiles(self) -> tuple[EnrollmentProfile, ...]:
        """Return completed participants with four enrollment WAVs."""

        try:
            profiles: list[EnrollmentProfile] = []
            with self._connect() as connection:
                participants = connection.execute(
                    """
                    SELECT p.telegram_user_id, c.participant_code
                    FROM telegram_participants p
                    JOIN participant_codes c
                        ON c.telegram_user_id = p.telegram_user_id
                    JOIN voice_samples v
                        ON v.telegram_user_id = p.telegram_user_id
                    WHERE p.accepted = 1
                    GROUP BY p.telegram_user_id, c.participant_code, p.required_samples
                    HAVING COUNT(v.sample_id) = p.required_samples
                    ORDER BY c.participant_code
                    """
                ).fetchall()
                for participant in participants:
                    rows = connection.execute(
                        """
                        SELECT prompt_index, wav_path
                        FROM voice_samples
                        WHERE telegram_user_id = ? AND prompt_index BETWEEN 1 AND 4
                        ORDER BY prompt_index
                        """,
                        (participant["telegram_user_id"],),
                    ).fetchall()
                    if len(rows) != 4:
                        continue
                    samples = tuple(
                        EnrollmentSample(
                            prompt_index=int(row["prompt_index"]),
                            wav_path=Path(str(row["wav_path"])),
                        )
                        for row in rows
                    )
                    if tuple(sample.prompt_index for sample in samples) != (1, 2, 3, 4):
                        continue
                    code = str(participant["participant_code"])
                    if not _participant_code_is_valid(code):
                        raise VoiceCollectionStoreError
                    profiles.append(
                        EnrollmentProfile(participant_code=code, samples=samples)
                    )
            return tuple(profiles)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            raise VoiceCollectionStoreError from exc

    def has_processed_message(self, *, telegram_user_id: int, message_id: int) -> bool:
        """Return whether a Telegram voice message was already handled."""

        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT 1 FROM processed_voice_messages
                    WHERE telegram_user_id = ? AND message_id = ?
                    """,
                    (telegram_user_id, message_id),
                ).fetchone()
            return row is not None
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            raise VoiceCollectionStoreError from exc

    def has_processed_identification_message(
        self, *, telegram_user_id: int, message_id: int
    ) -> bool:
        """Return whether an operator identification query was already handled."""

        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT 1 FROM processed_identification_messages
                    WHERE telegram_user_id = ? AND message_id = ?
                    """,
                    (telegram_user_id, message_id),
                ).fetchone()
            return row is not None
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            raise VoiceCollectionStoreError from exc

    def mark_processed_identification_message(
        self,
        *,
        telegram_user_id: int,
        message_id: int,
        update_id: int | None,
    ) -> None:
        """Persist an operator identification query replay marker."""

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO processed_identification_messages (
                        telegram_user_id, message_id, update_id
                    )
                    VALUES (?, ?, ?)
                    """,
                    (telegram_user_id, message_id, update_id),
                )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            raise VoiceCollectionStoreError from exc

    def commit_sample(
        self,
        *,
        telegram_user_id: int,
        reserved: ReservedSample,
        message_id: int,
        update_id: int | None,
    ) -> tuple[SessionSnapshot, bool]:
        """Persist a converted sample after the WAV has been verified."""

        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT subject_id, accepted, required_samples
                    FROM telegram_participants
                    WHERE telegram_user_id = ?
                    """,
                    (telegram_user_id,),
                ).fetchone()
                if row is None or int(row["accepted"]) != 1:
                    raise VoiceCollectionStoreError
                if self._message_exists(
                    connection,
                    telegram_user_id=telegram_user_id,
                    message_id=message_id,
                ):
                    return self._snapshot_from_connection(
                        connection,
                        telegram_user_id=telegram_user_id,
                    ), False
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
                        str(row["subject_id"]),
                        reserved.source_group_id,
                        CalibrationPartition.CALIBRATION.value,
                        reserved.prompt_index,
                        str(reserved.ogg_path),
                        str(reserved.wav_path),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO processed_voice_messages (
                        telegram_user_id, message_id, update_id
                    )
                    VALUES (?, ?, ?)
                    """,
                    (telegram_user_id, message_id, update_id),
                )
                return self._snapshot_from_connection(
                    connection,
                    telegram_user_id=telegram_user_id,
                ), True
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except sqlite3.IntegrityError:
            return self.get_session(telegram_user_id) or SessionSnapshot(
                subject_id="",
                participant_code=None,
                accepted=False,
                completed_samples=0,
                required_samples=REQUIRED_RECORDINGS,
            ), False
        except Exception as exc:
            raise VoiceCollectionStoreError from exc

    def delete_user(self, telegram_user_id: int) -> None:
        """Delete local records and audio for one Telegram user."""

        try:
            self._rewrite_managed_manifests_excluding_user(telegram_user_id)
            file_paths, subject_ids = self._delete_targets_for_user(telegram_user_id)
            for path in file_paths:
                if path.exists():
                    path.unlink()
            for subject_id in subject_ids:
                self._remove_empty_subject_tree(subject_id)
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DELETE FROM processed_voice_messages WHERE telegram_user_id = ?",
                    (telegram_user_id,),
                )
                connection.execute(
                    """
                    DELETE FROM processed_identification_messages
                    WHERE telegram_user_id = ?
                    """,
                    (telegram_user_id,),
                )
                connection.execute(
                    "DELETE FROM voice_samples WHERE telegram_user_id = ?",
                    (telegram_user_id,),
                )
                connection.execute(
                    "DELETE FROM participant_codes WHERE telegram_user_id = ?",
                    (telegram_user_id,),
                )
                connection.execute(
                    "DELETE FROM telegram_participants WHERE telegram_user_id = ?",
                    (telegram_user_id,),
                )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            raise VoiceCollectionStoreError from None

    def write_managed_manifest(
        self,
        *,
        output_path: Path,
        payload: dict[str, object],
    ) -> Path:
        """Atomically write and register a managed manifest artifact."""

        try:
            managed_path = self._canonical_managed_manifest_path(output_path)
            _validate_manifest_payload(payload)
            _write_json_atomic(managed_path, payload)
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO managed_manifest_artifacts (manifest_path)
                    VALUES (?)
                    """,
                    (str(managed_path),),
                )
            return managed_path
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            raise VoiceCollectionStoreError from None

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

    def _delete_targets_for_user(
        self,
        telegram_user_id: int,
    ) -> tuple[tuple[Path, ...], tuple[str, ...]]:
        with self._connect() as connection:
            file_rows = connection.execute(
                """
                SELECT ogg_path, wav_path
                FROM voice_samples
                WHERE telegram_user_id = ?
                """,
                (telegram_user_id,),
            ).fetchall()
            subject_rows = connection.execute(
                """
                SELECT subject_id FROM telegram_participants WHERE telegram_user_id = ?
                UNION
                SELECT subject_id FROM voice_samples WHERE telegram_user_id = ?
                """,
                (telegram_user_id, telegram_user_id),
            ).fetchall()
        paths: list[Path] = []
        for row in file_rows:
            paths.append(Path(str(row["ogg_path"])))
            paths.append(Path(str(row["wav_path"])))
        return tuple(paths), tuple(str(row["subject_id"]) for row in subject_rows)

    def _managed_manifest_paths(self) -> tuple[Path, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT manifest_path
                FROM managed_manifest_artifacts
                ORDER BY manifest_path
                """
            ).fetchall()
        return tuple(
            self._canonical_managed_manifest_path(Path(str(row["manifest_path"])))
            for row in rows
        )

    def _rewrite_managed_manifests_excluding_user(self, telegram_user_id: int) -> None:
        _, subject_ids = self._delete_targets_for_user(telegram_user_id)
        if not subject_ids:
            return
        subject_set = frozenset(subject_ids)
        for path in self._managed_manifest_paths():
            _rewrite_manifest_without_subjects(path=path, subject_ids=subject_set)

    def _canonical_managed_manifest_path(self, output_path: Path) -> Path:
        manifest_dir = self._data_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        canonical_dir = manifest_dir.resolve(strict=True)
        candidate = output_path
        if not candidate.is_absolute():
            candidate = manifest_dir / candidate
        candidate_parent = candidate.parent.resolve(strict=False)
        if candidate_parent != canonical_dir:
            raise VoiceCollectionStoreError
        if candidate.name in {"", ".", ".."} or not candidate.name.endswith(
            ".manifest.json"
        ):
            raise VoiceCollectionStoreError
        canonical = candidate.resolve(strict=False)
        if canonical.parent != canonical_dir:
            raise VoiceCollectionStoreError
        return canonical

    def _remove_empty_subject_tree(self, subject_id: str) -> None:
        subject_root = self._data_dir / "audio" / subject_id
        if not subject_root.exists():
            return
        for child in sorted(subject_root.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        subject_root.rmdir()

    def _message_exists(
        self,
        connection: sqlite3.Connection,
        *,
        telegram_user_id: int,
        message_id: int,
    ) -> bool:
        row = connection.execute(
            """
            SELECT 1 FROM processed_voice_messages
            WHERE telegram_user_id = ? AND message_id = ?
            """,
            (telegram_user_id, message_id),
        ).fetchone()
        return row is not None

    def _snapshot_from_connection(
        self,
        connection: sqlite3.Connection,
        *,
        telegram_user_id: int,
    ) -> SessionSnapshot:
        row = connection.execute(
            """
            SELECT subject_id, accepted, required_samples
            FROM telegram_participants
            WHERE telegram_user_id = ?
            """,
            (telegram_user_id,),
        ).fetchone()
        if row is None:
            raise VoiceCollectionStoreError
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
            participant_code=self._participant_code_from_connection(
                connection,
                telegram_user_id=telegram_user_id,
            ),
            accepted=bool(row["accepted"]),
            completed_samples=int(count_row[0]),
            required_samples=int(row["required_samples"]),
        )

    def _migrate_participant_codes(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT p.telegram_user_id, p.subject_id
            FROM telegram_participants p
            LEFT JOIN participant_codes c
                ON c.telegram_user_id = p.telegram_user_id
            WHERE c.telegram_user_id IS NULL
            ORDER BY p.rowid, p.subject_id
            """
        ).fetchall()
        for row in rows:
            self._assign_participant_code(
                connection,
                telegram_user_id=int(row["telegram_user_id"]),
                subject_id=str(row["subject_id"]),
            )

    def _assign_participant_code(
        self,
        connection: sqlite3.Connection,
        *,
        telegram_user_id: int,
        subject_id: str,
    ) -> str:
        existing = connection.execute(
            """
            SELECT participant_code
            FROM participant_codes
            WHERE telegram_user_id = ?
            """,
            (telegram_user_id,),
        ).fetchone()
        if existing is not None:
            code = str(existing["participant_code"])
            if not _participant_code_is_valid(code):
                raise VoiceCollectionStoreError
            return code
        used_rows = connection.execute(
            """
            SELECT participant_code
            FROM participant_code_reservations
            ORDER BY participant_code
            """
        ).fetchall()
        used = {str(row["participant_code"]) for row in used_rows}
        for number in range(1, _MAX_PARTICIPANT_CODE + 1):
            code = f"P{number:04d}"
            if code in used:
                continue
            connection.execute(
                """
                INSERT INTO participant_code_reservations (participant_code)
                VALUES (?)
                """,
                (code,),
            )
            connection.execute(
                """
                INSERT INTO participant_codes (
                    telegram_user_id, subject_id, participant_code
                )
                VALUES (?, ?, ?)
                """,
                (telegram_user_id, subject_id, code),
            )
            return code
        raise VoiceCollectionStoreError

    def _participant_code_from_connection(
        self,
        connection: sqlite3.Connection,
        *,
        telegram_user_id: int,
    ) -> str | None:
        row = connection.execute(
            """
            SELECT participant_code
            FROM participant_codes
            WHERE telegram_user_id = ?
            """,
            (telegram_user_id,),
        ).fetchone()
        if row is None:
            return None
        code = str(row["participant_code"])
        if not _participant_code_is_valid(code):
            raise VoiceCollectionStoreError
        return code

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _rewrite_manifest_without_subjects(
    *,
    path: Path,
    subject_ids: frozenset[str],
) -> None:
    temp_path: Path | None = None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if type(payload) is not dict:
            raise VoiceCollectionStoreError
        samples = payload.get("samples")
        if type(samples) is not list:
            raise VoiceCollectionStoreError
        rewritten_samples = []
        for sample in samples:
            if type(sample) is not dict:
                raise VoiceCollectionStoreError
            subject_id = sample.get("subject_id")
            if type(subject_id) is str and subject_id in subject_ids:
                continue
            rewritten_samples.append(dict(sample))
        rewritten = dict(payload)
        rewritten["samples"] = rewritten_samples
        _validate_manifest_payload(rewritten)
        temp_path = _write_json_atomic(path, rewritten)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        if temp_path is not None:
            _unlink_temp_manifest(temp_path)
        raise VoiceCollectionStoreError from None


def _validate_manifest_payload(payload: dict[str, object]) -> None:
    if type(payload) is not dict:
        raise VoiceCollectionStoreError
    if not set(payload).issubset({"repository_commit_sha", "samples", "thresholds"}):
        raise VoiceCollectionStoreError
    repository_commit_sha = payload.get("repository_commit_sha")
    samples = payload.get("samples")
    if type(repository_commit_sha) is not str or len(repository_commit_sha) != 40:
        raise VoiceCollectionStoreError
    if type(samples) is not list:
        raise VoiceCollectionStoreError
    for sample in samples:
        if type(sample) is not dict:
            raise VoiceCollectionStoreError
        if set(sample) != {
            "sample_id",
            "subject_id",
            "source_group_id",
            "partition",
            "wav_path",
        }:
            raise VoiceCollectionStoreError
        if type(sample.get("wav_path")) is not str or not sample.get("wav_path"):
            raise VoiceCollectionStoreError
        CalibrationSampleRecord(
            sample_id=sample["sample_id"],
            subject_id=sample["subject_id"],
            source_group_id=sample["source_group_id"],
            partition=CalibrationPartition(sample["partition"]),
        )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(payload, temp_file, ensure_ascii=False, indent=2, sort_keys=True)
            temp_file.write("\n")
            temp_file.flush()
        if temp_path is None:
            raise VoiceCollectionStoreError
        temp_path.replace(path)
        return temp_path
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        if temp_path is not None:
            _unlink_temp_manifest(temp_path)
        raise VoiceCollectionStoreError from None


def _unlink_temp_manifest(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _participant_code_is_valid(code: str) -> bool:
    return type(code) is str and _PARTICIPANT_CODE_RE.fullmatch(code) is not None

"""Local audio conversion for Telegram OGG/Opus voice messages."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Final

from voiceid.telegram_bot.config import FFMPEG_TIMEOUT_SECONDS, MAX_VOICE_SECONDS

TARGET_SAMPLE_RATE_HZ: Final = 16000


class AudioConversionError(ValueError):
    """Stable, privacy-safe audio conversion failure."""

    def __init__(self) -> None:
        super().__init__("Audio conversion failed.")


def convert_ogg_to_wav(
    *,
    source_ogg: Path,
    target_wav: Path,
    ffmpeg_path: str = "ffmpeg",
    timeout_seconds: float = FFMPEG_TIMEOUT_SECONDS,
) -> None:
    """Convert a Telegram OGG/Opus file to mono PCM16 WAV at 16 kHz."""

    target_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [
                ffmpeg_path,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_ogg),
                "-ac",
                "1",
                "-ar",
                str(TARGET_SAMPLE_RATE_HZ),
                "-c:a",
                "pcm_s16le",
                str(target_wav),
            ],
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception as exc:
        raise AudioConversionError from exc

    if completed.returncode != 0:
        _unlink_safely(target_wav)
        raise AudioConversionError
    _validate_wav(target_wav)


def _validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception as exc:
        raise AudioConversionError from exc

    if (
        channels != 1
        or sample_width != 2
        or sample_rate != TARGET_SAMPLE_RATE_HZ
        or frames <= 0
        or frames > TARGET_SAMPLE_RATE_HZ * MAX_VOICE_SECONDS
    ):
        _unlink_safely(path)
        raise AudioConversionError


def _unlink_safely(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass

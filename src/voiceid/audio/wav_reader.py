"""Low-level RIFF/WAVE inspection and PCM16 signal statistics."""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

PCM_FORMAT_TAG = 1
IEEE_FLOAT_FORMAT_TAG = 3
PCM16_ABS_MAX = 32768.0
PUBLIC_FLOAT_DECIMALS = 6

_RIFF_HEADER_SIZE = 12
_CHUNK_HEADER_SIZE = 8
_FMT_CHUNK_MIN_SIZE = 16
_PCM16_SAMPLE_WIDTH_BYTES = 2
_READ_CHUNK_FRAMES = 65536


class WavReadError(Exception):
    """Base class for controlled WAV read failures."""


class WavContainerError(WavReadError):
    """Raised when a file is not a RIFF/WAVE container."""


class WavDecodeError(WavReadError):
    """Raised when a RIFF/WAVE file cannot be safely decoded."""


class WavFileNotReadableError(WavReadError):
    """Raised when the file cannot be opened for reading."""


@dataclass(frozen=True, slots=True)
class WavHeader:
    """Parsed RIFF/WAVE header fields required by validation."""

    format_tag: int
    channels: int
    sample_rate_hz: int
    byte_rate: int
    block_align: int
    sample_width_bits: int
    data_size_bytes: int
    total_frames: int

    @property
    def codec(self) -> str:
        """Return a stable public codec name."""

        if self.format_tag == PCM_FORMAT_TAG and self.sample_width_bits == 16:
            return "PCM_S16LE"
        if self.format_tag == PCM_FORMAT_TAG:
            return f"PCM_{self.sample_width_bits}_BIT"
        if self.format_tag == IEEE_FLOAT_FORMAT_TAG:
            return "IEEE_FLOAT"
        return f"WAV_FORMAT_{self.format_tag}"


@dataclass(frozen=True, slots=True)
class PcmSignalStats:
    """Simple signal-level statistics for decoded PCM16 samples."""

    peak_amplitude: float
    rms_level: float
    clipped_sample_fraction: float


def read_wav_header(path: Path) -> WavHeader:
    """Parse a local RIFF/WAVE header without trusting the file extension."""

    try:
        file_size = path.stat().st_size
        with path.open("rb") as wav_file:
            riff_header = wav_file.read(_RIFF_HEADER_SIZE)
            if len(riff_header) < _RIFF_HEADER_SIZE:
                raise WavContainerError

            if riff_header[:4] != b"RIFF" or riff_header[8:12] != b"WAVE":
                raise WavContainerError

            fmt_fields: tuple[int, int, int, int, int, int] | None = None
            data_size_bytes: int | None = None

            while wav_file.tell() < file_size:
                chunk_header = wav_file.read(_CHUNK_HEADER_SIZE)
                if not chunk_header:
                    break
                if len(chunk_header) < _CHUNK_HEADER_SIZE:
                    raise WavDecodeError

                chunk_id = chunk_header[:4]
                chunk_size = struct.unpack("<I", chunk_header[4:8])[0]
                chunk_start = wav_file.tell()
                chunk_end = chunk_start + chunk_size
                if chunk_end > file_size:
                    raise WavDecodeError

                if chunk_id == b"fmt ":
                    chunk_data = wav_file.read(chunk_size)
                    if len(chunk_data) != chunk_size:
                        raise WavDecodeError
                    fmt_fields = _parse_fmt_chunk(chunk_data)
                elif chunk_id == b"data":
                    data_size_bytes = chunk_size
                    wav_file.seek(chunk_end)
                else:
                    wav_file.seek(chunk_end)

                if chunk_size % 2:
                    if wav_file.tell() + 1 > file_size:
                        raise WavDecodeError
                    wav_file.seek(1, 1)

            if fmt_fields is None or data_size_bytes is None:
                raise WavDecodeError

            format_tag, channels, sample_rate, byte_rate, block_align, bits = fmt_fields
            if block_align <= 0 or data_size_bytes % block_align != 0:
                raise WavDecodeError

            return WavHeader(
                format_tag=format_tag,
                channels=channels,
                sample_rate_hz=sample_rate,
                byte_rate=byte_rate,
                block_align=block_align,
                sample_width_bits=bits,
                data_size_bytes=data_size_bytes,
                total_frames=data_size_bytes // block_align,
            )
    except PermissionError as exc:
        raise WavFileNotReadableError from exc
    except OSError as exc:
        raise WavDecodeError from exc


def decode_pcm16_signal_stats(
    path: Path,
    header: WavHeader,
    *,
    clipping_sample_level_threshold: float,
) -> PcmSignalStats:
    """Decode PCM16 samples in chunks and calculate normalized signal stats."""

    try:
        with wave.open(str(path), "rb") as wav_file:
            _assert_wave_matches_header(wav_file, header)

            scalar_count = 0
            frames_read = 0
            square_sum = 0.0
            peak_amplitude = 0.0
            clipped_samples = 0

            while frames_read < header.total_frames:
                frames_to_read = min(
                    _READ_CHUNK_FRAMES,
                    header.total_frames - frames_read,
                )
                chunk = wav_file.readframes(frames_to_read)
                if not chunk:
                    break
                if len(chunk) % _PCM16_SAMPLE_WIDTH_BYTES != 0:
                    raise WavDecodeError

                value_count = len(chunk) // _PCM16_SAMPLE_WIDTH_BYTES
                if value_count % header.channels != 0:
                    raise WavDecodeError

                frames_read += value_count // header.channels
                scalar_count += value_count

                for (sample,) in struct.iter_unpack("<h", chunk):
                    normalized_sample = sample / PCM16_ABS_MAX
                    abs_sample = abs(normalized_sample)
                    square_sum += normalized_sample * normalized_sample
                    peak_amplitude = max(peak_amplitude, abs_sample)
                    if abs_sample >= clipping_sample_level_threshold:
                        clipped_samples += 1

            expected_scalar_count = header.total_frames * header.channels
            if (
                frames_read != header.total_frames
                or scalar_count != expected_scalar_count
            ):
                raise WavDecodeError
            if scalar_count == 0:
                raise WavDecodeError

            rms_level = math.sqrt(square_sum / scalar_count)
            clipped_fraction = clipped_samples / scalar_count
            return PcmSignalStats(
                peak_amplitude=round(peak_amplitude, PUBLIC_FLOAT_DECIMALS),
                rms_level=round(rms_level, PUBLIC_FLOAT_DECIMALS),
                clipped_sample_fraction=round(clipped_fraction, PUBLIC_FLOAT_DECIMALS),
            )
    except PermissionError as exc:
        raise WavFileNotReadableError from exc
    except (EOFError, OSError, struct.error, wave.Error) as exc:
        raise WavDecodeError from exc


def calculate_duration_seconds(header: WavHeader) -> float | None:
    """Return duration as total frames per channel divided by sample rate."""

    if header.sample_rate_hz <= 0:
        return None
    duration_seconds = header.total_frames / header.sample_rate_hz
    return round(duration_seconds, PUBLIC_FLOAT_DECIMALS)


def _parse_fmt_chunk(
    chunk_data: bytes,
) -> tuple[int, int, int, int, int, int]:
    if len(chunk_data) < _FMT_CHUNK_MIN_SIZE:
        raise WavDecodeError
    return struct.unpack("<HHIIHH", chunk_data[:_FMT_CHUNK_MIN_SIZE])


def _assert_wave_matches_header(wav_file: wave.Wave_read, header: WavHeader) -> None:
    if wav_file.getcomptype() != "NONE":
        raise WavDecodeError
    if wav_file.getnchannels() != header.channels:
        raise WavDecodeError
    if wav_file.getframerate() != header.sample_rate_hz:
        raise WavDecodeError
    if wav_file.getsampwidth() != _PCM16_SAMPLE_WIDTH_BYTES:
        raise WavDecodeError
    if wav_file.getnframes() != header.total_frames:
        raise WavDecodeError

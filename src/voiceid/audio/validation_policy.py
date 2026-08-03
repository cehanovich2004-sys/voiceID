"""Validation policy constants for Phase 2 WAV loading."""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_SAMPLE_RATES_HZ = frozenset({8000, 16000, 22050, 44100, 48000})
SUPPORTED_CHANNEL_COUNTS = frozenset({1, 2})
TARGET_SAMPLE_RATE_HZ = 16000
TARGET_CHANNEL_COUNT = 1
REQUIRED_SAMPLE_WIDTH_BITS = 16

DEFAULT_MIN_DURATION_SECONDS = 1.0
DEFAULT_MAX_DURATION_SECONDS = 60.0

SILENT_RMS_THRESHOLD = 0.0001
LOW_AUDIO_RMS_THRESHOLD = 0.01
CLIPPING_PEAK_THRESHOLD = 0.9999
CLIPPING_SAMPLE_LEVEL_THRESHOLD = 0.999
CLIPPING_SAMPLE_FRACTION_THRESHOLD = 0.001


@dataclass(frozen=True, slots=True)
class AudioValidationPolicy:
    """Configurable technical validation policy for local WAV files."""

    min_duration_seconds: float = DEFAULT_MIN_DURATION_SECONDS
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS
    supported_sample_rates_hz: frozenset[int] = SUPPORTED_SAMPLE_RATES_HZ
    supported_channel_counts: frozenset[int] = SUPPORTED_CHANNEL_COUNTS
    target_sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ
    target_channel_count: int = TARGET_CHANNEL_COUNT
    required_sample_width_bits: int = REQUIRED_SAMPLE_WIDTH_BITS
    silent_rms_threshold: float = SILENT_RMS_THRESHOLD
    low_audio_rms_threshold: float = LOW_AUDIO_RMS_THRESHOLD
    clipping_peak_threshold: float = CLIPPING_PEAK_THRESHOLD
    clipping_sample_level_threshold: float = CLIPPING_SAMPLE_LEVEL_THRESHOLD
    clipping_sample_fraction_threshold: float = CLIPPING_SAMPLE_FRACTION_THRESHOLD

    def __post_init__(self) -> None:
        """Validate policy consistency early."""

        if self.min_duration_seconds <= 0:
            msg = "min_duration_seconds must be greater than zero"
            raise ValueError(msg)
        if self.max_duration_seconds < self.min_duration_seconds:
            msg = "max_duration_seconds must be greater than or equal to minimum"
            raise ValueError(msg)
        if not self.supported_sample_rates_hz:
            msg = "supported_sample_rates_hz must not be empty"
            raise ValueError(msg)
        if not self.supported_channel_counts:
            msg = "supported_channel_counts must not be empty"
            raise ValueError(msg)
        if self.required_sample_width_bits <= 0:
            msg = "required_sample_width_bits must be greater than zero"
            raise ValueError(msg)
        if self.silent_rms_threshold < 0:
            msg = "silent_rms_threshold must not be negative"
            raise ValueError(msg)
        if self.low_audio_rms_threshold <= self.silent_rms_threshold:
            msg = "low_audio_rms_threshold must be greater than silent threshold"
            raise ValueError(msg)
        if not 0 < self.clipping_peak_threshold <= 1:
            msg = "clipping_peak_threshold must be in the range (0, 1]"
            raise ValueError(msg)
        if not 0 < self.clipping_sample_level_threshold <= 1:
            msg = "clipping_sample_level_threshold must be in the range (0, 1]"
            raise ValueError(msg)
        if not 0 <= self.clipping_sample_fraction_threshold <= 1:
            msg = "clipping_sample_fraction_threshold must be in the range [0, 1]"
            raise ValueError(msg)

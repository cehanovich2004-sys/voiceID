"""Tests for Phase 3 deterministic audio preprocessing."""

from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from voiceid.audio.preprocessing import (
    TARGET_PREPROCESSING_SAMPLE_RATE_HZ,
    Float32Waveform,
    PreprocessedAudioMetadata,
    PreprocessedAudioResult,
    PreprocessingErrorCode,
    PreprocessingStatus,
)
from voiceid.audio.validation_policy import AudioValidationPolicy
from voiceid.services import preprocess_wav_file

FloatArray = npt.NDArray[np.float32]


def test_mono_16khz_pcm16_returns_float32_mono_without_resampling(
    tmp_path: Path,
) -> None:
    source = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.4)
    audio_path = _write_waveform(tmp_path / "mono-16k.wav", source, 16000)

    result = preprocess_wav_file(audio_path)

    waveform = _valid_waveform(result)
    metadata = _valid_metadata(result)
    expected = _remove_mean(_writer_decoded_float(source))
    assert waveform.dtype == np.dtype(np.float32)
    assert waveform.shape == (16000,)
    assert metadata.source_sample_rate_hz == 16000
    assert metadata.source_channels == 1
    assert metadata.output_sample_rate_hz == TARGET_PREPROCESSING_SAMPLE_RATE_HZ
    assert metadata.output_channels == 1
    assert metadata.output_samples == 16000
    assert metadata.downmixed_to_mono is False
    assert metadata.resampled is False
    assert metadata.resample_up == 1
    assert metadata.resample_down == 1
    np.testing.assert_allclose(waveform, expected, atol=1e-6)


def test_stereo_downmix_uses_arithmetic_mean_in_float_domain(
    tmp_path: Path,
) -> None:
    left = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.5)
    right = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.25)
    stereo = _stereo(left, right)
    audio_path = _write_waveform(tmp_path / "stereo-mean.wav", stereo, 16000)

    result = preprocess_wav_file(audio_path)

    waveform = _valid_waveform(result)
    metadata = _valid_metadata(result)
    decoded = _writer_decoded_float(stereo)
    expected = _remove_mean(np.mean(decoded, axis=1, dtype=np.float32))
    assert metadata.downmixed_to_mono is True
    np.testing.assert_allclose(waveform, expected, atol=1e-6)


def test_stereo_identical_channels_match_mono_result(tmp_path: Path) -> None:
    source = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.35)
    mono_path = _write_waveform(tmp_path / "mono.wav", source, 16000)
    stereo_path = _write_waveform(
        tmp_path / "stereo.wav", _stereo(source, source), 16000
    )

    mono_result = preprocess_wav_file(mono_path)
    stereo_result = preprocess_wav_file(stereo_path)

    np.testing.assert_allclose(
        _valid_waveform(stereo_result),
        _valid_waveform(mono_result),
        atol=1e-6,
    )
    assert _valid_metadata(stereo_result).downmixed_to_mono is True


def test_stereo_channels_with_different_levels_are_averaged(
    tmp_path: Path,
) -> None:
    left = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.6)
    right = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.2)
    audio_path = _write_waveform(
        tmp_path / "different-levels.wav", _stereo(left, right), 16000
    )

    result = preprocess_wav_file(audio_path)

    waveform = _valid_waveform(result)
    peak = float(np.max(np.abs(waveform)))
    assert 0.39 <= peak <= 0.41


def test_stereo_antiphase_channels_cancel_after_downmix(tmp_path: Path) -> None:
    source = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.4)
    audio_path = _write_waveform(
        tmp_path / "antiphase.wav", _stereo(source, -source), 16000
    )

    result = preprocess_wav_file(audio_path)

    waveform = _valid_waveform(result)
    assert np.max(np.abs(waveform)) <= 1e-6


def test_valid_antiphase_stereo_can_return_zero_waveform_by_contract(
    tmp_path: Path,
) -> None:
    source = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.4)
    audio_path = _write_waveform(
        tmp_path / "antiphase-contract.wav", _stereo(source, -source), 16000
    )

    result = preprocess_wav_file(audio_path)

    waveform = _valid_waveform(result)
    assert _valid_metadata(result).downmixed_to_mono is True
    assert np.max(np.abs(waveform)) <= 1e-6


def test_valid_constant_signal_can_return_zero_waveform_after_dc_removal(
    tmp_path: Path,
) -> None:
    source = np.full(16000, 4 / 32768, dtype=np.float32)
    audio_path = _write_waveform(tmp_path / "constant-valid.wav", source, 16000)

    result = preprocess_wav_file(audio_path)

    waveform = _valid_waveform(result)
    assert _valid_metadata(result).dc_offset_removed is True
    assert np.max(np.abs(waveform)) <= 1e-6


@pytest.mark.parametrize("sample_rate_hz", [8000, 22050, 44100, 48000])
def test_supported_source_rates_are_resampled_to_16khz(
    tmp_path: Path,
    sample_rate_hz: int,
) -> None:
    source = _sine_wave(
        sample_rate_hz=sample_rate_hz,
        frequency_hz=440,
        amplitude=0.35,
    )
    audio_path = _write_waveform(
        tmp_path / f"rate-{sample_rate_hz}.wav",
        source,
        sample_rate_hz,
    )

    result = preprocess_wav_file(audio_path)

    waveform = _valid_waveform(result)
    metadata = _valid_metadata(result)
    rate_gcd = math.gcd(sample_rate_hz, TARGET_PREPROCESSING_SAMPLE_RATE_HZ)
    assert waveform.shape == (TARGET_PREPROCESSING_SAMPLE_RATE_HZ,)
    assert metadata.output_sample_rate_hz == TARGET_PREPROCESSING_SAMPLE_RATE_HZ
    assert metadata.output_samples == TARGET_PREPROCESSING_SAMPLE_RATE_HZ
    assert metadata.resampled is True
    assert metadata.resample_up == TARGET_PREPROCESSING_SAMPLE_RATE_HZ // rate_gcd
    assert metadata.resample_down == sample_rate_hz // rate_gcd


def test_resampled_length_and_duration_stay_consistent(tmp_path: Path) -> None:
    sample_rate_hz = 22050
    duration_seconds = 1.25
    source = _sine_wave(
        sample_rate_hz=sample_rate_hz,
        frequency_hz=440,
        amplitude=0.35,
        duration_seconds=duration_seconds,
    )
    audio_path = _write_waveform(tmp_path / "duration.wav", source, sample_rate_hz)

    result = preprocess_wav_file(audio_path)

    metadata = _valid_metadata(result)
    expected_duration = round(
        metadata.output_samples / TARGET_PREPROCESSING_SAMPLE_RATE_HZ,
        6,
    )
    assert metadata.output_duration_seconds == expected_duration
    assert abs(metadata.output_duration_seconds - metadata.source_duration_seconds) <= (
        1 / TARGET_PREPROCESSING_SAMPLE_RATE_HZ
    )


def test_resampling_suppresses_energy_above_target_nyquist(
    tmp_path: Path,
) -> None:
    low_frequency = _sine_wave(
        sample_rate_hz=48000,
        frequency_hz=1000,
        amplitude=0.5,
    )
    high_frequency = _sine_wave(
        sample_rate_hz=48000,
        frequency_hz=12000,
        amplitude=0.5,
    )
    low_path = _write_waveform(tmp_path / "low.wav", low_frequency, 48000)
    high_path = _write_waveform(tmp_path / "high.wav", high_frequency, 48000)

    low_result = preprocess_wav_file(low_path)
    high_result = preprocess_wav_file(high_path)

    low_rms = _rms(_valid_waveform(low_result))
    high_rms = _rms(_valid_waveform(high_result))
    assert high_rms < low_rms * 0.2


def test_dc_offset_is_removed_after_decode(tmp_path: Path) -> None:
    source = _sine_wave(
        sample_rate_hz=16000,
        frequency_hz=440,
        amplitude=0.2,
    ) + np.float32(0.25)
    audio_path = _write_waveform(tmp_path / "dc-offset.wav", source, 16000)

    result = preprocess_wav_file(audio_path)

    waveform = _valid_waveform(result)
    assert abs(float(np.mean(waveform, dtype=np.float64))) < 1e-6
    assert _valid_metadata(result).dc_offset_removed is True


@pytest.mark.parametrize("sample_value", [0.0, 1 / 32768.0])
def test_silence_and_near_silence_fail_before_preprocessing(
    tmp_path: Path,
    sample_value: float,
) -> None:
    source = np.full(16000, sample_value, dtype=np.float32)
    audio_path = _write_waveform(tmp_path / f"silent-{sample_value}.wav", source, 16000)

    result = preprocess_wav_file(audio_path)

    _assert_invalid_without_partial_result(result, PreprocessingErrorCode.INVALID_INPUT)


def test_safety_clipping_protects_against_internal_overshoot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.35)
    audio_path = _write_waveform(tmp_path / "overshoot.wav", source, 16000)

    def fake_resample_to_target(
        _waveform: Float32Waveform,
        _source_sample_rate_hz: int,
    ) -> tuple[Float32Waveform, bool, int, int]:
        return np.array([-1.2, 0.0, 1.2], dtype=np.float32), True, 1, 1

    monkeypatch.setattr(
        "voiceid.audio.preprocessing._resample_to_target",
        fake_resample_to_target,
    )

    result = preprocess_wav_file(audio_path)

    waveform = _valid_waveform(result)
    metadata = _valid_metadata(result)
    np.testing.assert_array_equal(
        waveform,
        np.array([-1.0, 0.0, 1.0], dtype=np.float32),
    )
    assert metadata.safety_clipped is True


@pytest.mark.parametrize(
    ("duration_seconds", "expected_samples"),
    [(1.0, 16000), (60.0, 960000)],
)
def test_minimum_and_maximum_supported_durations_preprocess(
    tmp_path: Path,
    duration_seconds: float,
    expected_samples: int,
) -> None:
    source = _sine_wave(
        sample_rate_hz=16000,
        frequency_hz=440,
        amplitude=0.3,
        duration_seconds=duration_seconds,
    )
    audio_path = _write_waveform(
        tmp_path / f"duration-{duration_seconds}.wav", source, 16000
    )

    result = preprocess_wav_file(audio_path)

    assert _valid_waveform(result).shape == (expected_samples,)
    assert _valid_metadata(result).output_duration_seconds == duration_seconds


def test_repeated_preprocessing_is_deterministic(tmp_path: Path) -> None:
    source = _sine_wave(sample_rate_hz=44100, frequency_hz=440, amplitude=0.35)
    audio_path = _write_waveform(tmp_path / "deterministic.wav", source, 44100)

    first = preprocess_wav_file(audio_path)
    second = preprocess_wav_file(audio_path)

    assert np.array_equal(_valid_waveform(first), _valid_waveform(second))
    assert _valid_metadata(first) == _valid_metadata(second)


def test_output_invariants_are_enforced(tmp_path: Path) -> None:
    source = _sine_wave(sample_rate_hz=22050, frequency_hz=440, amplitude=0.4)
    audio_path = _write_waveform(tmp_path / "invariants.wav", source, 22050)

    result = preprocess_wav_file(audio_path)

    waveform = _valid_waveform(result)
    assert waveform.dtype == np.dtype(np.float32)
    assert waveform.ndim == 1
    assert waveform.shape[0] > 0
    assert np.all(np.isfinite(waveform))
    assert float(np.min(waveform)) >= -1.0
    assert float(np.max(waveform)) <= 1.0


def test_preprocessing_does_not_modify_input_file(tmp_path: Path) -> None:
    source = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.35)
    audio_path = _write_waveform(tmp_path / "immutable-input.wav", source, 16000)
    before_hash = _sha256(audio_path)

    _ = preprocess_wav_file(audio_path)

    assert _sha256(audio_path) == before_hash


def test_preprocessing_does_not_create_additional_audio_or_temp_files(
    tmp_path: Path,
) -> None:
    source = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.35)
    audio_path = _write_waveform(tmp_path / "single-input.wav", source, 16000)
    before_entries = {entry.name for entry in tmp_path.iterdir()}

    _ = preprocess_wav_file(audio_path)

    assert {entry.name for entry in tmp_path.iterdir()} == before_entries


def test_public_result_does_not_leak_path_or_waveform_values(tmp_path: Path) -> None:
    source = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.35)
    audio_path = _write_waveform(tmp_path / "privacy.wav", source, 16000)

    result = preprocess_wav_file(audio_path)
    payload = result.to_dict()
    public_text = f"{result!r} {payload!r}"

    assert _valid_waveform(result).size == 16000
    assert str(tmp_path) not in public_text
    assert "array(" not in public_text
    assert "waveform" not in repr(result)
    assert "waveform" not in payload
    assert payload["file_name"] == "privacy.wav"


def test_invalid_wav_returns_safe_result_without_partial_waveform(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "not-wav.wav"
    audio_path.write_bytes(b"not a wav")

    result = preprocess_wav_file(audio_path)

    _assert_invalid_without_partial_result(result, PreprocessingErrorCode.INVALID_INPUT)
    assert str(tmp_path) not in repr(result.to_dict())


def test_unexpected_exception_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.35)
    audio_path = _write_waveform(tmp_path / "unexpected.wav", source, 16000)

    def raise_unexpected_error(*_args: object, **_kwargs: object) -> Float32Waveform:
        msg = f"decoder failed at {audio_path.resolve()} with {np.array([0.1])}"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        "voiceid.services.audio_preprocessing.decode_pcm16_to_float32_from_file",
        raise_unexpected_error,
    )

    result = preprocess_wav_file(audio_path)
    public_text = f"{result!r} {result.to_dict()!r} {caplog.text}"

    _assert_invalid_without_partial_result(
        result,
        PreprocessingErrorCode.PREPROCESSING_ERROR,
    )
    assert str(audio_path.resolve()) not in public_text
    assert "array(" not in public_text


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_base_exceptions_are_not_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exc_type: type[BaseException],
) -> None:
    source = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.35)
    audio_path = _write_waveform(tmp_path / f"{exc_type.__name__}.wav", source, 16000)

    def raise_base_exception(*_args: object, **_kwargs: object) -> Float32Waveform:
        raise exc_type

    monkeypatch.setattr(
        "voiceid.services.audio_preprocessing.decode_pcm16_to_float32_from_file",
        raise_base_exception,
    )

    with pytest.raises(exc_type):
        preprocess_wav_file(audio_path)


def test_public_dict_contract_is_stable(tmp_path: Path) -> None:
    source = _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.35)
    audio_path = _write_waveform(tmp_path / "contract.wav", source, 16000)

    result = preprocess_wav_file(audio_path)
    payload = result.to_dict()

    assert payload == {
        "status": "VALID",
        "file_name": "contract.wav",
        "metadata": _valid_metadata(result).to_dict(),
        "errors": [],
    }


@pytest.mark.parametrize(
    "replacement",
    [
        {"sample_rate_hz": 16000, "channels": 1, "duration_seconds": 0.5},
        {"sample_rate_hz": 16000, "channels": 1, "duration_seconds": 61.0},
        {"sample_rate_hz": 48000, "channels": 2, "duration_seconds": 1.25},
    ],
)
def test_replacement_after_snapshot_validation_cannot_mix_metadata_and_waveform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: dict[str, int | float],
) -> None:
    audio_path = _write_waveform(
        tmp_path / "race.wav",
        _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.35),
        16000,
    )
    replacement_path = tmp_path / "replacement.wav"
    _write_replacement_waveform(replacement_path, replacement)
    original_validate = _service_function("validate_wav_signal_snapshot")

    def replace_after_validation(*args: object, **kwargs: object) -> object:
        result = original_validate(*args, **kwargs)
        replacement_path.replace(audio_path)
        return result

    monkeypatch.setattr(
        "voiceid.services.audio_preprocessing.validate_wav_signal_snapshot",
        replace_after_validation,
    )

    result = preprocess_wav_file(audio_path)

    metadata = _valid_metadata(result)
    assert _valid_waveform(result).shape == (16000,)
    _assert_metadata_matches_original_snapshot(metadata)
    _assert_metadata_is_internally_consistent(metadata)


def test_replacement_before_header_read_still_uses_open_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = _write_waveform(
        tmp_path / "header-race.wav",
        _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.35),
        16000,
    )
    replacement_path = _write_waveform(
        tmp_path / "header-race-replacement.wav",
        _sine_wave(
            sample_rate_hz=48000,
            frequency_hz=440,
            amplitude=0.35,
            duration_seconds=1.25,
        ),
        48000,
    )
    original_read_header = _service_function("read_wav_header_from_file")

    def replace_before_header(*args: object, **kwargs: object) -> object:
        replacement_path.replace(audio_path)
        return original_read_header(*args, **kwargs)

    monkeypatch.setattr(
        "voiceid.services.audio_preprocessing.read_wav_header_from_file",
        replace_before_header,
    )

    result = preprocess_wav_file(audio_path)

    metadata = _valid_metadata(result)
    _assert_metadata_matches_original_snapshot(metadata)
    _assert_metadata_is_internally_consistent(metadata)


def test_replacement_before_decode_still_uses_open_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = _write_waveform(
        tmp_path / "decode-race.wav",
        _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.35),
        16000,
    )
    replacement_path = _write_waveform(
        tmp_path / "decode-race-replacement.wav",
        _sine_wave(
            sample_rate_hz=16000,
            frequency_hz=440,
            amplitude=0.35,
            duration_seconds=61.0,
        ),
        16000,
    )
    original_decode = _service_function("decode_pcm16_to_float32_from_file")

    def replace_before_decode(*args: object, **kwargs: object) -> object:
        replacement_path.replace(audio_path)
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(
        "voiceid.services.audio_preprocessing.decode_pcm16_to_float32_from_file",
        replace_before_decode,
    )

    result = preprocess_wav_file(audio_path)

    metadata = _valid_metadata(result)
    assert _valid_waveform(result).shape == (16000,)
    _assert_metadata_matches_original_snapshot(metadata)
    _assert_metadata_is_internally_consistent(metadata)


@pytest.mark.parametrize("duration_seconds", [0.5, 61.0])
def test_actual_snapshot_policy_violation_is_invalid_without_partial_result(
    tmp_path: Path,
    duration_seconds: float,
) -> None:
    source = _sine_wave(
        sample_rate_hz=16000,
        frequency_hz=440,
        amplitude=0.35,
        duration_seconds=duration_seconds,
    )
    audio_path = _write_waveform(tmp_path / "policy-violation.wav", source, 16000)

    result = preprocess_wav_file(audio_path)

    _assert_invalid_without_partial_result(result, PreprocessingErrorCode.INVALID_INPUT)


def test_custom_policy_applies_to_actual_snapshot(
    tmp_path: Path,
) -> None:
    source = _sine_wave(
        sample_rate_hz=16000,
        frequency_hz=440,
        amplitude=0.35,
        duration_seconds=1.25,
    )
    audio_path = _write_waveform(tmp_path / "custom-policy.wav", source, 16000)

    result = preprocess_wav_file(
        audio_path,
        policy=AudioValidationPolicy(max_duration_seconds=1.0),
    )

    _assert_invalid_without_partial_result(result, PreprocessingErrorCode.INVALID_INPUT)


def test_truncated_actual_snapshot_returns_controlled_decode_error_without_leaks(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "truncated.wav"
    audio_path.write_bytes(b"RIFF\x20\x00\x00\x00WAVEfmt ")

    result = preprocess_wav_file(audio_path)
    public_text = f"{result!r} {result.to_dict()!r}"

    _assert_invalid_without_partial_result(result, PreprocessingErrorCode.DECODE_ERROR)
    assert str(audio_path.resolve()) not in public_text
    assert "array(" not in public_text
    assert "waveform" not in repr(result)


def test_replacement_with_truncated_file_after_validation_does_not_crash_or_mix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = _write_waveform(
        tmp_path / "valid-then-truncated.wav",
        _sine_wave(sample_rate_hz=16000, frequency_hz=440, amplitude=0.35),
        16000,
    )
    truncated_path = tmp_path / "truncated-replacement.wav"
    truncated_path.write_bytes(b"RIFF\x20\x00\x00\x00WAVEfmt ")
    original_validate = _service_function("validate_wav_signal_snapshot")

    def replace_after_validation(*args: object, **kwargs: object) -> object:
        result = original_validate(*args, **kwargs)
        truncated_path.replace(audio_path)
        return result

    monkeypatch.setattr(
        "voiceid.services.audio_preprocessing.validate_wav_signal_snapshot",
        replace_after_validation,
    )

    result = preprocess_wav_file(audio_path)

    metadata = _valid_metadata(result)
    _assert_metadata_matches_original_snapshot(metadata)
    _assert_metadata_is_internally_consistent(metadata)


def _valid_waveform(result: PreprocessedAudioResult) -> Float32Waveform:
    assert result.status == PreprocessingStatus.VALID
    assert result.is_valid is True
    assert result.errors == ()
    assert result.waveform is not None
    return result.waveform


def _valid_metadata(result: PreprocessedAudioResult) -> PreprocessedAudioMetadata:
    assert result.metadata is not None
    return result.metadata


def _assert_invalid_without_partial_result(
    result: PreprocessedAudioResult,
    code: PreprocessingErrorCode,
) -> None:
    assert result.status == PreprocessingStatus.INVALID
    assert result.is_valid is False
    assert result.waveform is None
    assert result.metadata is None
    assert [error.code for error in result.errors] == [code.value]


def _assert_metadata_matches_original_snapshot(
    metadata: PreprocessedAudioMetadata,
) -> None:
    assert metadata.source_sample_rate_hz == 16000
    assert metadata.source_channels == 1
    assert metadata.source_duration_seconds == 1.0
    assert metadata.output_samples == 16000
    assert metadata.output_duration_seconds == 1.0


def _assert_metadata_is_internally_consistent(
    metadata: PreprocessedAudioMetadata,
) -> None:
    assert metadata.output_sample_rate_hz == TARGET_PREPROCESSING_SAMPLE_RATE_HZ
    assert metadata.output_channels == 1
    assert metadata.output_duration_seconds == round(
        metadata.output_samples / TARGET_PREPROCESSING_SAMPLE_RATE_HZ,
        6,
    )


def _service_function(name: str) -> object:
    import voiceid.services.audio_preprocessing as audio_preprocessing

    return getattr(audio_preprocessing, name)


def _sine_wave(
    *,
    sample_rate_hz: int,
    frequency_hz: float,
    amplitude: float,
    duration_seconds: float = 1.0,
) -> FloatArray:
    sample_count = int(round(sample_rate_hz * duration_seconds))
    timeline = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
    waveform = amplitude * np.sin(2 * np.pi * frequency_hz * timeline)
    return cast(FloatArray, waveform.astype(np.float32))


def _stereo(left: FloatArray, right: FloatArray) -> FloatArray:
    return cast(FloatArray, np.column_stack((left, right)).astype(np.float32))


def _write_waveform(path: Path, waveform: FloatArray, sample_rate_hz: int) -> Path:
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim == 1:
        channels = 1
    elif values.ndim == 2:
        channels = int(values.shape[1])
    else:
        raise ValueError("test helper supports only mono or multi-channel arrays")

    max_positive = np.float32(32767 / 32768)
    pcm = np.rint(np.clip(values, -1.0, max_positive) * 32768.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm.tobytes())
    return path


def _write_replacement_waveform(
    path: Path,
    replacement: dict[str, int | float],
) -> Path:
    sample_rate_hz = int(replacement["sample_rate_hz"])
    channels = int(replacement["channels"])
    duration_seconds = float(replacement["duration_seconds"])
    source = _sine_wave(
        sample_rate_hz=sample_rate_hz,
        frequency_hz=440,
        amplitude=0.35,
        duration_seconds=duration_seconds,
    )
    if channels == 2:
        source = _stereo(source, source)
    return _write_waveform(path, source, sample_rate_hz)


def _writer_decoded_float(waveform: FloatArray) -> FloatArray:
    max_positive = np.float32(32767 / 32768)
    pcm = np.rint(np.clip(waveform, -1.0, max_positive) * 32768.0).astype("<i2")
    return cast(FloatArray, pcm.astype(np.float32) / np.float32(32768.0))


def _remove_mean(waveform: FloatArray) -> FloatArray:
    mean_value = np.mean(waveform, dtype=np.float64)
    return cast(
        FloatArray,
        (waveform.astype(np.float32, copy=False) - np.float32(mean_value)).astype(
            np.float32,
            copy=False,
        ),
    )


def _rms(waveform: FloatArray) -> float:
    squares = np.square(waveform.astype(np.float64))
    return float(np.sqrt(np.mean(squares)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

# Phase 2: WAV Loading And Technical Validation

Status: implemented in feature branch, pending CTO review.

Phase 2 validates local WAV files against the technical input contract for the
MVP audio pipeline. Technical validation does not guarantee suitability for
speaker verification.

Phase 2 does not perform ML inference, speaker embeddings, scoring, downmixing,
resampling, normalization, VAD, denoising, diarization, anti-spoofing, speech
recognition, or storage of audio files.

Real voice recordings, embeddings, model weights, and datasets must not be
committed to Git.

## Public Application API

Use the application service:

```python
from voiceid.services import validate_wav_file

result = validate_wav_file("sample.wav")
payload = result.to_dict()
```

The result exposes only a safe filename. It does not expose canonical or
absolute local paths.

## Validation Statuses

- `VALID`: no hard errors and no warnings.
- `VALID_WITH_WARNINGS`: no hard errors, but one or more warnings.
- `INVALID`: one or more hard errors.

Warnings do not make the file invalid. Any hard error makes the file invalid.

## Supported WAV Format

A file is technically supported only when all of these conditions are true:

- container: `RIFF/WAVE`;
- codec: uncompressed PCM;
- sample width: 16 bit;
- channels: mono or stereo;
- sample rate: 8000, 16000, 22050, 44100, or 48000 Hz;
- duration: from 1 to configured maximum seconds, inclusive;
- default maximum duration: 60 seconds;
- decoded waveform has at least one sample;
- decoded waveform is not fully or practically silent.

Boundary durations of exactly 1 second and exactly 60 seconds are accepted by
the default policy.

The future downstream ML target is mono 16000 Hz. Phase 2 uses that target only
for warnings:

- stereo input returns `STEREO_AUDIO`;
- sample rate other than 16000 Hz returns `SAMPLE_RATE_NOT_TARGET`.

Phase 2 does not convert stereo to mono and does not resample audio.

## Result Contract

`AudioValidationResult` contains:

- `status`: `VALID`, `VALID_WITH_WARNINGS`, or `INVALID`;
- `file_name`: sanitized basename only;
- `metadata`: `AudioMetadata` or empty public metadata for failures before
  safe parsing;
- `warnings`: stable issue objects;
- `errors`: stable issue objects.

Issue objects contain at least:

- `code`;
- `message`.

They may also contain safe structured details:

- `field`;
- `measured_value`;
- `expected`.

For `INVALID` results, metadata may be empty or partially populated depending
on how much information was safely parsed before the hard error.

## Metadata Definitions

- `container`: `WAV` when the RIFF/WAVE container is safely detected.
- `codec`: stable codec name, such as `PCM_S16LE`.
- `sample_rate_hz`: WAV sample rate from the header.
- `channels`: WAV channel count from the header.
- `sample_width_bits`: WAV bit depth from the header.
- `duration_seconds`: `total_samples / sample_rate_hz`, rounded to 6 decimal
  places.
- `total_samples`: number of audio frames per channel, not the total count of
  interleaved scalar values.
- `peak_amplitude`: maximum absolute PCM16 sample value normalized by 32768.0
  into the 0.0-1.0 range, rounded to 6 decimal places.
- `rms_level`: square root of the mean squared normalized PCM16 scalar samples,
  calculated across all channels, rounded to 6 decimal places.
- `resampling_required`: true when sample rate differs from the future 16000 Hz
  target.
- `mono_conversion_required`: true when channel count differs from the future
  mono target.

Stereo peak and RMS are calculated across the interleaved scalar samples from
both channels. The original audio is not modified.

## Error Codes

- `FILE_NOT_FOUND`
- `FILE_NOT_READABLE`
- `FILE_EMPTY`
- `UNSUPPORTED_EXTENSION`
- `INVALID_WAV_CONTAINER`
- `UNSUPPORTED_WAV_CODEC`
- `UNSUPPORTED_SAMPLE_WIDTH`
- `UNSUPPORTED_SAMPLE_RATE`
- `UNSUPPORTED_CHANNEL_COUNT`
- `ZERO_SAMPLES`
- `DURATION_TOO_SHORT`
- `DURATION_TOO_LONG`
- `SILENT_AUDIO`
- `DECODE_ERROR`

User-facing errors are controlled and stable. Internal decoder exceptions and
absolute filesystem paths are not exposed in the public result.

## Warning Codes

- `SAMPLE_RATE_NOT_TARGET`
- `STEREO_AUDIO`
- `LOW_AUDIO_LEVEL`
- `POSSIBLE_CLIPPING`

Warnings are deterministic technical heuristics. They are not claims about
speaker-verification quality.

## Configurable Policy

`AudioValidationPolicy` controls the technical validation limits.

Defaults:

- `min_duration_seconds = 1.0`
- `max_duration_seconds = 60.0`
- `supported_sample_rates_hz = {8000, 16000, 22050, 44100, 48000}`
- `supported_channel_counts = {1, 2}`
- `target_sample_rate_hz = 16000`
- `target_channel_count = 1`
- `required_sample_width_bits = 16`

Signal-level heuristic thresholds:

- `silent_rms_threshold = 0.0001`
- `low_audio_rms_threshold = 0.01`
- `clipping_peak_threshold = 0.9999`
- `clipping_sample_level_threshold = 0.999`
- `clipping_sample_fraction_threshold = 0.001`

Silence policy:

- fully zero waveform: `INVALID`, `SILENT_AUDIO`;
- RMS at or below `0.0001`: `INVALID`, `SILENT_AUDIO`;
- RMS below `0.01` but above the silence threshold: `VALID_WITH_WARNINGS`,
  `LOW_AUDIO_LEVEL`.

Clipping policy:

- peak amplitude at or above `0.9999`: `POSSIBLE_CLIPPING`;
- or at least `0.1%` of scalar samples at or above `0.999`: `POSSIBLE_CLIPPING`.

These thresholds are conservative technical heuristics for Phase 2. They do not
confirm speech presence, speaker count, noise level, or suitability for speaker
verification.

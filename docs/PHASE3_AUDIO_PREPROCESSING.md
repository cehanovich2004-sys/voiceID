# Phase 3: Deterministic Audio Preprocessing

Status: accepted and merged.

Phase 3 prepares Phase 2-valid WAV files for future speaker embedding
experiments. It converts supported PCM16 WAV input into deterministic float32
mono 16000 Hz waveform data.

Technical validation does not guarantee suitability for speaker verification.
Phase 3 preprocessing also does not guarantee suitability for speaker
verification.

Real voice recordings, embeddings, model weights, datasets, and generated audio
fixtures must not be committed to Git.

## Public Application API

Use the application service:

```python
from voiceid.services import preprocess_wav_file

result = preprocess_wav_file("sample.wav")
if result.is_valid:
    waveform = result.waveform
    metadata = result.metadata
payload = result.to_dict()
```

`waveform` is available only as an in-memory `numpy.ndarray` attribute on a
valid result. The public dictionary and representation exclude waveform values.
The result exposes only a safe filename and never exposes canonical or absolute
local paths.

## Pipeline

The deterministic Phase 3 pipeline is:

1. Open the local WAV file once.
2. Read and validate the RIFF/WAVE header from that open file snapshot.
3. Decode signal statistics from the same open file snapshot.
4. Apply the Phase 2 validation policy to the same snapshot metadata and
   signal statistics.
5. Reject invalid Phase 2 input without returning a partial waveform.
6. Decode PCM16 samples from the same open file snapshot without integer
   overflow.
7. Convert samples to float32 using `sample / 32768.0`.
8. Downmix stereo with an arithmetic mean in floating point.
9. Skip downmix for mono input.
10. Remove DC offset by subtracting the waveform mean.
11. Resample to 16000 Hz with `scipy.signal.resample_poly`.
12. Skip resampling when the source sample rate is already 16000 Hz.
13. Reduce `up` and `down` resampling factors with `gcd(input_rate, 16000)`.
14. Apply safety clipping to `[-1.0, 1.0]`.
15. Validate output invariants before returning a valid result.

Safety clipping is only a guard against numerical overshoot from processing.
It is not peak, gain, RMS, loudness, or speech normalization.

Validation, source metadata, and PCM decode are intentionally tied to the same
open file snapshot. If the pathname is replaced after the file is opened, the
current preprocessing operation continues to use the already opened snapshot
instead of mixing metadata from one file state with waveform data from another.

## Result Contract

`PreprocessedAudioResult` contains:

- `status`: `VALID` or `INVALID`;
- `file_name`: sanitized basename only;
- `waveform`: `numpy.ndarray` with `dtype=float32`, one-dimensional mono shape,
  and sample rate 16000 Hz when valid, otherwise `None`;
- `metadata`: `PreprocessedAudioMetadata` when valid, otherwise `None`;
- `errors`: stable issue objects for invalid results.

`to_dict()` contains only:

- `status`;
- `file_name`;
- `metadata`;
- `errors`.

It intentionally does not include the waveform. `repr(result)`, user-facing
errors, and normal logs must not include waveform values or filesystem paths.

## Metadata Definitions

- `source_sample_rate_hz`: original WAV sample rate from Phase 2-valid input.
- `source_channels`: original WAV channel count.
- `source_duration_seconds`: Phase 2 duration in seconds.
- `output_sample_rate_hz`: always `16000`.
- `output_channels`: always `1`.
- `output_samples`: number of samples in the returned mono waveform.
- `output_duration_seconds`: `output_samples / 16000`, rounded to 6 decimals.
- `downmixed_to_mono`: true only for stereo input.
- `dc_offset_removed`: true for every valid result.
- `resampled`: true when source sample rate differs from 16000 Hz.
- `resample_up`: reduced numerator passed to `resample_poly`.
- `resample_down`: reduced denominator passed to `resample_poly`.
- `safety_clipped`: true when post-processing values exceeded `[-1.0, 1.0]`
  before the final guard.

The returned waveform must be finite, one-dimensional, `float32`, and clipped
to `[-1.0, 1.0]`.

`VALID` means the preprocessing contract succeeded. It does not mean that the
post-preprocessing signal is useful for speaker verification. For example,
antiphase stereo can become zero after arithmetic mean downmix, and a technically
valid near-constant signal can become zero after DC offset removal. A
post-preprocessing energy policy is deferred to a future CTO decision.

## Error Codes

- `INVALID_INPUT`: Phase 2 technical validation failed.
- `DECODE_ERROR`: Phase 2-valid input could not be decoded or checked safely
  during preprocessing.
- `PREPROCESSING_ERROR`: an unexpected internal exception was converted into a
  controlled public failure.

`KeyboardInterrupt` and `SystemExit` are not caught or sanitized.

## Dependencies

Phase 3 adds runtime dependencies:

- `numpy>=1.26,<3`
- `scipy>=1.11,<2`

`numpy` is used for deterministic float32 array operations. `scipy` is used for
polyphase sample-rate conversion through `scipy.signal.resample_poly`.

## Non-goals

Phase 3 does not implement:

- VAD;
- silence trimming;
- denoising;
- peak, gain, RMS, loudness, or speech normalization;
- speaker embeddings;
- similarity scoring;
- probability or match percentage;
- `MATCH`, `NO_MATCH`, or `UNCERTAIN`;
- API endpoints;
- UI;
- database storage;
- saving or copying audio files.
- post-preprocessing energy acceptance decisions.

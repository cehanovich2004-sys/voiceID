# Phase 4: Baseline Speaker Embeddings

Status: accepted and merged in Phase 4B.

Phase 4B adds a local, typed, privacy-safe speaker embedding layer. It consumes
only Phase 3 `PreprocessedAudioResult` objects and produces a validated
embedding result.

Phase 4B does not compare two embeddings, calculate cosine similarity, return a
probability, choose thresholds, or produce `MATCH`, `NO_MATCH`, or `UNCERTAIN`.

## Baseline Model

The approved baseline backend is:

- model identifier: `speechbrain/spkrec-ecapa-voxceleb`;
- pinned revision: `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`;
- expected embedding dimension: `192`;
- inference device: CPU;
- normalized output: `False`.

The backend uses SpeechBrain tensor-only `encode_batch(..., normalize=False)`.
It does not call `classify_file()` or `verify_files()`, does not create a
temporary WAV, and does not perform similarity scoring.

## Dependency Boundary

SpeechBrain and PyTorch are optional dependencies:

```bash
python -m pip install -e ".[dev,embeddings]"
```

The core package, fake backend tests, and normal quality gates do not require
SpeechBrain, PyTorch, model weights, or network access. Real-model smoke tests
require both `VOICEID_RUN_REAL_MODEL_TESTS=1` and
`VOICEID_SPEECHBRAIN_ECAPA_CACHE_DIR`.

Candidate pins verified during Phase 4B dependency spike:

- Python `3.11.15`;
- `speechbrain==1.1.0`;
- `torch==2.8.0`;
- `torchaudio==2.8.0`;
- `huggingface_hub>=0.24,<1`;
- NumPy `2.4.6`;
- SciPy `1.17.1`.

The spike produced raw SpeechBrain output shape `[1, 1, 192]`, flattened by the
adapter to public shape `[192]` with `dtype=float32` and finite values.

## Cache And Offline Runtime

Runtime should use a prepared local cache in strict offline mode. The explicit
bootstrap/download mode downloads only the required model files:

- `hyperparams.yaml`;
- `embedding_model.ckpt`;
- `mean_var_norm_emb.ckpt`;
- `classifier.ckpt`;
- `label_encoder.txt`;
- `config.json`.

Example WAV/FLAC files from the model repository are intentionally excluded from
the expected snapshot manifest.

Model weights and Hugging Face cache directories must not be committed to Git.
Missing or incomplete cache is converted into stable public error codes without
exposing absolute paths.

## Public Application API

```python
from pathlib import Path

from voiceid.embeddings.backends.speechbrain_ecapa import (
    SpeechBrainEcapaBackendFactory,
    default_speechbrain_ecapa_config,
)
from voiceid.embeddings.loader import EmbeddingModelLoader
from voiceid.services import preprocess_wav_file
from voiceid.services.speaker_embedding import SpeakerEmbeddingService

preprocessed = preprocess_wav_file("sample.wav")
config = default_speechbrain_ecapa_config(
    cache_dir=Path("local-model-cache"),
    offline=True,
)
service = SpeakerEmbeddingService(
    loader=EmbeddingModelLoader(SpeechBrainEcapaBackendFactory(config)),
)
embedding_result = service.embed(preprocessed)
```

The application service:

- accepts only `PreprocessedAudioResult`;
- never accepts or returns a path;
- never reads WAV files;
- never resamples, downmixes, or normalizes audio;
- does not know SpeechBrain tensor shapes;
- uses a backend protocol and model loader boundary.

## Result Contract

`SpeakerEmbeddingResult` contains:

- `status`: `VALID` or `INVALID`;
- `embedding`: one-dimensional `numpy.ndarray` with `dtype=float32` and shape
  `[192]` only when valid, otherwise `None`;
- `metadata`: `EmbeddingMetadata` only when valid, otherwise `None`;
- `errors`: stable issue objects for invalid results.

Metadata contains only non-secret fields:

- `embedding_dimension`;
- `model_identifier`;
- `model_revision`;
- `backend_name`;
- `device`;
- `input_sample_rate_hz`;
- `input_samples`;
- `input_duration_seconds`;
- `normalized`.

`to_dict()`, `repr()`, `str()`, user-facing errors, and normal logs must not
include waveform values, embedding values, absolute paths, cache paths, access
tokens, tensor contents, or raw exception text.

## Input Validation

The service validates the Phase 3 object before backend inference:

- status is valid;
- waveform and metadata are present;
- sample rate is 16000 Hz;
- waveform has `dtype=float32`, one dimension, finite values, non-empty shape,
  and range `[-1.0, 1.0]`;
- metadata length and duration match the waveform.

The input waveform is copied before backend inference and must not be modified.

## Zero Or Near-Zero Policy

Phase 3 can validly produce a zero waveform after antiphase stereo downmix or
DC offset removal. Phase 4B rejects zero or near-zero input before model
inference with:

```text
ZERO_OR_NEAR_ZERO_WAVEFORM
```

The deterministic RMS policy is:

```text
rms = sqrt(mean(square(waveform.astype(float64))))
invalid when rms <= 1e-8
```

This is a degenerate-signal guard, not VAD and not a decision that speech is or
is not present. Low-energy signals above the threshold are not automatically
rejected.

## Error Codes

- `INVALID_PREPROCESSED_AUDIO`
- `UNSUPPORTED_SAMPLE_RATE`
- `EMPTY_WAVEFORM`
- `NON_FINITE_WAVEFORM`
- `ZERO_OR_NEAR_ZERO_WAVEFORM`
- `MODEL_NOT_LOADED`
- `MODEL_LOAD_FAILED`
- `MODEL_CACHE_MISSING`
- `MODEL_CACHE_CORRUPTED`
- `INFERENCE_FAILED`
- `INVALID_EMBEDDING_SHAPE`
- `INVALID_EMBEDDING_DTYPE`
- `NON_FINITE_EMBEDDING`
- `MEMORY_LIMIT_EXCEEDED`

`KeyboardInterrupt` and `SystemExit` are not caught or sanitized.

## Privacy And Security

Speaker embeddings are sensitive biometric templates. They must not be logged,
serialized in public responses, committed to Git, or treated as anonymous
technical metadata.

Inference is local. Runtime can be offline after explicit cache preparation.
Commercial production use requires separate legal review of model weights,
training data, consent, retention, and biometric-processing obligations.

## Non-goals

Phase 4B does not implement:

- cosine similarity;
- probability or match percentage;
- thresholds;
- `MATCH`, `NO_MATCH`, or `UNCERTAIN`;
- enrollment or registration;
- VAD;
- diarization;
- one-to-many identification;
- anti-spoofing;
- API, UI, or database storage.

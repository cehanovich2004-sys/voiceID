# VoiceID

VoiceID is an MVP-stage project for voice verification research.

The product goal is to answer one question:

> Do two audio recordings belong to the same person?

Phase 4 adds a local baseline speaker embedding layer for the MVP audio
pipeline. Similarity scoring, API endpoints, and user interfaces are
intentionally deferred to later phases.

## Current Status

Status: Phase 4B, baseline speaker embedding layer in review.

Implemented:

- `src`-layout Python package named `voiceid`;
- minimal typed configuration;
- standard-library logging setup;
- typed WAV validation contract;
- local RIFF/WAVE PCM16 metadata validation;
- deterministic warning and error codes;
- deterministic PCM16 WAV preprocessing to in-memory float32 mono 16000 Hz;
- typed, privacy-safe speaker embedding contract;
- optional SpeechBrain ECAPA-TDNN backend integration;
- smoke tests;
- linting, formatting, type checking, and CI setup;
- documentation, ADR-001, ADR-002, ADR-003, and ADR-004.

Not implemented yet:

- similarity engine;
- API;
- Streamlit UI.

VoiceID is not a production-ready biometric identification system. Any future
biometric processing must be designed with security, privacy, consent,
retention, and legal requirements in mind.

## Python Version

Use Python 3.11.

The project intentionally targets `>=3.11,<3.12` for Phase 1 to keep the base
runtime conservative for future ML dependencies such as PyTorch, SpeechBrain,
pyannote.audio, librosa, and soundfile.

## Installation

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the project with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the optional embedding backend only when working on Phase 4 model
inference:

```bash
python -m pip install -e ".[dev,embeddings]"
```

Verify that the package imports:

```bash
python -c "import voiceid; print(voiceid.__version__)"
```

## WAV Validation

Validate a local WAV file through the application service:

```python
from voiceid.services import validate_wav_file

result = validate_wav_file("sample.wav")
print(result.to_dict())
```

Phase 2 supports only RIFF/WAVE PCM16 files with mono or stereo audio, sample
rate 8000, 16000, 22050, 44100, or 48000 Hz, and duration from 1 to 60 seconds
by default.

Technical validation does not guarantee suitability for speaker verification.
The result never exposes the canonical or absolute local path; only a safe
filename is returned.

## Audio Preprocessing

Preprocess a Phase 2-valid local WAV file through the application service:

```python
from voiceid.services import preprocess_wav_file

result = preprocess_wav_file("sample.wav")
if result.is_valid:
    waveform = result.waveform
    metadata = result.metadata
```

Phase 3 converts valid PCM16 WAV input to an in-memory `numpy.ndarray` with
`dtype=float32`, mono shape, sample rate 16000 Hz, finite values, and range
`[-1.0, 1.0]`.

The pipeline is deterministic: PCM16 decode, `sample / 32768.0`, stereo
arithmetic mean downmix, DC offset removal, `scipy.signal.resample_poly`
resampling when needed, and safety clipping. Safety clipping is a final
invariant guard, not audio normalization.

The public result does not expose absolute paths or waveform values in
`repr()`, `to_dict()`, normal logs, or user-facing errors.

Phase 3 ties header validation, source metadata, signal statistics, and PCM
decode to one open file snapshot. `VALID` means the preprocessing contract
succeeded; it does not mean the resulting signal is useful for speaker
verification. Antiphase stereo or near-constant input can validly produce a
zero waveform after downmix or DC offset removal.

## Speaker Embeddings

Extract a baseline speaker embedding from a valid Phase 3 result through the
application service:

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
result = service.embed(preprocessed)
```

Phase 4B uses `speechbrain/spkrec-ecapa-voxceleb` pinned to revision
`0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`. The backend accepts only the
Phase 3 contract: in-memory mono `float32` waveform, sample rate 16000 Hz, and
shape `[samples]`. It returns a raw, unnormalized 192-dimensional `float32`
embedding.

The first model cache preparation is an explicit bootstrap/download operation.
Runtime can then run in strict offline mode from a prepared local cache. Model
weights and Hugging Face cache files are not stored in Git.

`VALID` embedding means only that extraction succeeded. It does not mean
`MATCH`, does not compare two voices, and does not expose probability or a
threshold. Speaker embeddings are sensitive biometric templates and must not be
logged, serialized in public payloads, or committed to Git.

## Quality Checks

Run the same checks locally that CI runs:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

Real-model smoke tests are opt-in and require a prepared local SpeechBrain cache:

```bash
VOICEID_RUN_REAL_MODEL_TESTS=1 \
VOICEID_SPEECHBRAIN_ECAPA_CACHE_DIR=/path/to/cache \
pytest -m real_model
```

To apply Ruff formatting:

```bash
ruff format .
```

## Project Structure

```text
voiceID/
├── src/
│   └── voiceid/
│       ├── __init__.py
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── preprocessing.py
│       │   ├── validation_policy.py
│       │   └── wav_reader.py
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── contracts.py
│       │   ├── loader.py
│       │   ├── policy.py
│       │   └── backends/
│       │       ├── __init__.py
│       │       ├── base.py
│       │       └── speechbrain_ecapa.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── audio_preprocessing.py
│       │   ├── audio_validation.py
│       │   └── speaker_embedding.py
│       ├── config.py
│       ├── logging_config.py
│       └── py.typed
├── tests/
│   ├── __init__.py
│   ├── test_audio_preprocessing.py
│   ├── test_smoke.py
│   └── test_wav_validation.py
├── docs/
│   ├── ML_PHASE1_AUDIO_AND_BASELINE_RECOMMENDATIONS.md
│   ├── PHASE4_SPEAKER_EMBEDDINGS.md
│   ├── PHASE3_AUDIO_PREPROCESSING.md
│   ├── PHASE2_WAV_VALIDATION.md
│   ├── VOICEID_PROJECT_SPEC.md
│   ├── ROADMAP.md
│   ├── PROJECT_STATE.md
│   └── adr/
│       ├── README.md
│       ├── ADR-001-project-structure.md
│       ├── ADR-002-wav-validation.md
│       ├── ADR-003-deterministic-audio-preprocessing.md
│       └── ADR-004-baseline-embedding-backend.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── .editorconfig
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

## Branch And Pull Request Workflow

Work should happen on a dedicated branch, not directly on `main`.

Recommended flow:

```bash
git switch -c chore/short-description
ruff check .
ruff format --check .
mypy src
pytest
git push -u origin chore/short-description
```

Open a Pull Request into `main` and wait for technical review before merging.

Do not commit real voice recordings, speaker embeddings, model weights, model
caches, access tokens, or datasets to Git.

## Similarity Is Not Probability

Future phases may compute cosine similarity between speaker embeddings. That
score must not be presented as a match probability by default.

A probability-like value is only acceptable after experimental calibration on a
labeled dataset that covers same-speaker and different-speaker pairs.

# VoiceID

VoiceID is an MVP-stage project for voice verification research.

The product goal is to answer one question:

> Do two audio recordings belong to the same person?

Phase 2 adds local WAV loading and technical validation for the future MVP audio
pipeline. Voice embeddings, similarity scoring, API endpoints, and user
interfaces are intentionally deferred to later phases.

## Current Status

Status: Phase 2, WAV loading and technical validation.

Implemented:

- `src`-layout Python package named `voiceid`;
- minimal typed configuration;
- standard-library logging setup;
- typed WAV validation contract;
- local RIFF/WAVE PCM16 metadata validation;
- deterministic warning and error codes;
- smoke tests;
- linting, formatting, type checking, and CI setup;
- documentation, ADR-001, and ADR-002.

Not implemented yet:

- ML models;
- audio preprocessing and transforms;
- speaker embeddings;
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

## Quality Checks

Run the same checks locally that CI runs:

```bash
ruff check .
ruff format --check .
mypy src
pytest
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
│       │   ├── validation_policy.py
│       │   └── wav_reader.py
│       ├── services/
│       │   ├── __init__.py
│       │   └── audio_validation.py
│       ├── config.py
│       ├── logging_config.py
│       └── py.typed
├── tests/
│   ├── __init__.py
│   ├── test_smoke.py
│   └── test_wav_validation.py
├── docs/
│   ├── ML_PHASE1_AUDIO_AND_BASELINE_RECOMMENDATIONS.md
│   ├── PHASE2_WAV_VALIDATION.md
│   ├── VOICEID_PROJECT_SPEC.md
│   ├── ROADMAP.md
│   ├── PROJECT_STATE.md
│   └── adr/
│       ├── README.md
│       ├── ADR-001-project-structure.md
│       └── ADR-002-wav-validation.md
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

Do not commit real voice recordings, speaker embeddings, model weights, or
datasets to Git.

## Similarity Is Not Probability

Future phases may compute cosine similarity between speaker embeddings. That
score must not be presented as a match probability by default.

A probability-like value is only acceptable after experimental calibration on a
labeled dataset that covers same-speaker and different-speaker pairs.

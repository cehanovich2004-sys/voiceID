# VoiceID

VoiceID is an MVP-stage project for voice verification research.

The product goal is to answer one question:

> Do two audio recordings belong to the same person?

Phase 1 does not implement audio loading, voice embeddings, similarity scoring,
API endpoints, or a user interface. It only creates the Python project
foundation required for later MVP phases.

## Current Status

Status: Phase 1, project foundation.

Implemented:

- `src`-layout Python package named `voiceid`;
- minimal typed configuration;
- standard-library logging setup;
- smoke tests;
- linting, formatting, type checking, and CI setup;
- documentation and ADR-001.

Not implemented yet:

- ML models;
- audio processing;
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
│       ├── config.py
│       ├── logging_config.py
│       └── py.typed
├── tests/
│   ├── __init__.py
│   └── test_smoke.py
├── docs/
│   ├── VOICEID_PROJECT_SPEC.md
│   ├── ROADMAP.md
│   ├── PROJECT_STATE.md
│   └── adr/
│       ├── README.md
│       └── ADR-001-project-structure.md
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

## Similarity Is Not Probability

Future phases may compute cosine similarity between speaker embeddings. That
score must not be presented as a match probability by default.

A probability-like value is only acceptable after experimental calibration on a
labeled dataset that covers same-speaker and different-speaker pairs.

# VoiceID Roadmap

## Phase 1: Project Foundation

Goal: create an installable, testable Python project foundation.

Completion criteria:

- package `voiceid` can be installed and imported;
- minimal configuration and logging exist;
- `ruff`, `mypy`, and `pytest` pass;
- CI is configured;
- ADR-001 documents the foundation decisions.

## Phase 2: Audio Loading And Validation

Goal: load local audio files and validate basic file properties.

Completion criteria:

- WAV files can be read safely;
- invalid paths, unsupported formats, empty files, and unreadable files fail
  with clear errors;
- sample rate, duration, channel count, and basic metadata are returned;
- tests cover valid and invalid audio inputs.

## Phase 3: Audio Preprocessing

Goal: prepare audio for future embedding extraction.

Completion criteria:

- mono conversion is defined;
- target sample rate is selected;
- normalization behavior is implemented and tested;
- silence trimming or VAD strategy is documented before implementation.

## Phase 4: Speaker Embeddings

Goal: connect a pretrained speaker embedding backend.

Completion criteria:

- one backend is selected through an ADR;
- model loading is isolated from application services;
- embeddings are produced for validated audio;
- tests use mocks for unit coverage and a small real-audio smoke test where
  practical.

## Phase 5: Similarity Engine

Goal: compare embeddings and return a decision.

Completion criteria:

- cosine similarity is implemented;
- output distinguishes `MATCH`, `NO MATCH`, and `UNCERTAIN`;
- thresholds are documented as experimental defaults;
- documentation clearly states that similarity is not probability.

## Phase 6: MVP Interface

Goal: provide a simple user-facing MVP interface.

Completion criteria:

- two files can be selected;
- the system displays validation status and comparison output;
- user-facing language avoids unsupported biometric guarantees;
- no authentication, CRM, telephony, or production workflows are added.

## Phase 7: Experimental Quality Evaluation

Goal: evaluate MVP behavior on labeled audio pairs.

Completion criteria:

- evaluation dataset requirements are documented;
- same-speaker and different-speaker pairs are tested;
- noise, short clips, and recording-device differences are analyzed;
- calibration requirements for probability-like outputs are defined.

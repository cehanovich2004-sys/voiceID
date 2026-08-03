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

- RIFF/WAVE PCM16 files can be read safely;
- invalid paths, unsupported formats, empty files, unreadable files, and
  corrupted files fail with stable error codes;
- sample rate, duration, channel count, bit depth, peak, RMS, and total samples
  are returned as technical metadata;
- stereo and supported non-16 kHz files return warnings, not transforms;
- technical validation is documented as separate from speaker-verification
  suitability;
- tests cover valid, invalid, and warning scenarios without real voice data.

## Phase 3: Audio Preprocessing

Goal: prepare audio for future embedding extraction.

Completion criteria:

- Phase 2-valid PCM16 WAV files are decoded deterministically;
- PCM16 samples are converted to float32 with `sample / 32768.0`;
- stereo input is downmixed with arithmetic mean and mono input is unchanged;
- DC offset removal is implemented and tested;
- supported source rates are resampled to 16000 Hz with `resample_poly`;
- safety clipping is documented as an invariant guard, not normalization;
- result metadata and privacy constraints are documented and tested;
- VAD, silence trimming, denoising, embeddings, and scoring remain out of
  scope.

## Phase 4: Speaker Embeddings

Goal: connect a pretrained speaker embedding backend.

Completion criteria:

- one backend is selected through an ADR;
- model loading is isolated from application services;
- embeddings are produced from Phase 3 preprocessed audio;
- public contracts hide embedding values and paths;
- zero or near-zero preprocessed waveforms are rejected before model inference;
- tests use fake backends for unit coverage and opt-in real-model smoke tests
  with synthetic audio and prepared local cache;
- similarity, probability, thresholds, and verdicts remain out of scope.

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

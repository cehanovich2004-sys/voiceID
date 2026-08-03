# VoiceID Project State

## Current Phase

Phase 3: deterministic audio preprocessing.

Implementation status: in progress in feature branch, pending checks and Pull
Request review.

## Completed In Phase 1

- Created `src`-layout Python package.
- Added minimal application configuration.
- Added centralized standard-library logging setup.
- Added smoke tests.
- Added `pyproject.toml` with pytest, Ruff, and mypy configuration.
- Added GitHub Actions CI.
- Added README, roadmap, project state, and ADR-001.
- Moved the project specification into `docs/`.

## Completed In Phase 2

- Added typed validation result models with stable statuses, warnings, and
  errors.
- Added configurable `AudioValidationPolicy` for WAV technical limits and
  deterministic signal-level heuristics.
- Added RIFF/WAVE header inspection and PCM16 chunked signal statistics using
  the Python standard library.
- Added application service `validate_wav_file()`.
- Added synthetic WAV tests for valid, invalid, and warning scenarios.
- Added Phase 2 validation documentation and ADR-002.

## Completed In Phase 3

- Added deterministic preprocessing from Phase 2-valid PCM16 WAV input to
  float32 mono 16000 Hz waveform output.
- Added `numpy` and `scipy` runtime dependencies for array processing and
  polyphase resampling.
- Added typed preprocessing result models and application service
  `preprocess_wav_file()`.
- Added synthetic preprocessing tests for downmixing, resampling, DC offset
  removal, safety clipping, deterministic output, and privacy guardrails.
- Added Phase 3 preprocessing documentation and ADR-003.

## Assumptions

- Python 3.11 is the safest Phase 1 runtime target for future ML dependency
  compatibility.
- The `>=3.11,<3.12` runtime range is a temporary Phase 1 constraint, not a
  long-term product requirement. Compatibility will be reviewed before adding
  audio and ML dependencies.
- The MVP should remain a modular monolith until real scaling pressure exists.
- The original project specification is retained as product context, while
  Phase 1 uses `src`-layout instead of the earlier draft `app/` layout.
- No audio, ML, API, UI, or database logic belongs in Phase 1.
- Phase 3 preprocessing accepts Phase 2-valid input, including files that are
  valid with warnings. It does not decide speaker-verification suitability.

## Open Questions

- Should Phase 2 remain WAV-only after CTO review, or should additional local
  formats be considered in a later phase?
- Should the conservative low-level and clipping heuristic thresholds be tuned
  after collecting non-sensitive test samples?
- What labeled evaluation data can be used later for similarity threshold
  calibration?
- Should future ML backend requirements change the target preprocessing
  contract beyond mono 16000 Hz float32?
- Should Phase 4 keep waveform arrays in memory only, or introduce explicit
  temporary storage with a separate privacy and retention decision?

## Known Risks

- Voice verification is biometric functionality and will require careful
  privacy, security, consent, and retention decisions before real use.
- Future ML dependencies may impose stricter Python, OS, or hardware
  constraints.
- Cosine similarity can be useful for comparison but is not a probability
  without calibration on labeled data.
- Short, noisy, or low-quality recordings may produce misleading results in
  later phases.
- Technical validation only confirms container and signal-level constraints. It
  does not confirm speech presence, speaker count, authenticity, or suitability
  for speaker verification.
- Deterministic preprocessing can make audio structurally compatible with a
  future embedding backend, but it does not prove that enough usable speech is
  present.

## Next Step

Open Phase 3 Pull Request for CTO review. Do not start Phase 4 until Phase 3 is
reviewed and merged.

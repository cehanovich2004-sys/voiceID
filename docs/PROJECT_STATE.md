# VoiceID Project State

## Current Phase

Phase 1: project foundation.

## Completed In This Phase

- Created `src`-layout Python package.
- Added minimal application configuration.
- Added centralized standard-library logging setup.
- Added smoke tests.
- Added `pyproject.toml` with pytest, Ruff, and mypy configuration.
- Added GitHub Actions CI.
- Added README, roadmap, project state, and ADR-001.
- Moved the project specification into `docs/`.

## Assumptions

- Python 3.11 is the safest Phase 1 runtime target for future ML dependency
  compatibility.
- The MVP should remain a modular monolith until real scaling pressure exists.
- The original project specification is retained as product context, while
  Phase 1 uses `src`-layout instead of the earlier draft `app/` layout.
- No audio, ML, API, UI, or database logic belongs in Phase 1.

## Open Questions

- Which audio formats should Phase 2 accept beyond WAV, if any?
- What minimum audio duration should be considered valid?
- Should Phase 2 normalize all inputs to 16 kHz immediately or only report
  metadata first?
- What labeled evaluation data can be used later for similarity threshold
  calibration?

## Known Risks

- Voice verification is biometric functionality and will require careful
  privacy, security, consent, and retention decisions before real use.
- Future ML dependencies may impose stricter Python, OS, or hardware
  constraints.
- Cosine similarity can be useful for comparison but is not a probability
  without calibration on labeled data.
- Short, noisy, or low-quality recordings may produce misleading results in
  later phases.

## Next Step

Phase 2: implement audio file loading, metadata extraction, and validation.

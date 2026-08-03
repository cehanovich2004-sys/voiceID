# VoiceID Project State

## Current Phase

No implementation phase is currently active. Phase 5A completed independent
QA and CTO review and is merged into `main`. Phase 5B has not started and
requires separate CTO approval.

## Completed Phases

- Phase 1 established the installable `src`-layout package, quality gates, CI,
  and architecture documentation.
- Phase 2 added safe RIFF/WAVE PCM16 loading and technical validation.
- Phase 3 added deterministic preprocessing to float32 mono 16000 Hz waveform.
- Phase 4B added the optional local SpeechBrain ECAPA-TDNN embedding backend,
  typed privacy-safe contracts, offline cache handling, and fake/real-model
  tests.
- Phase 5A added deterministic, privacy-safe raw cosine similarity for
  compatible speaker embeddings without a biometric decision policy.

## Phase 5A Completed

- Added a pure `compare_speaker_embeddings()` application API.
- Added typed `VALID`/`INVALID` similarity contracts with one stable error.
- Added float64 cosine calculation and numeric overshoot clipping.
- Added fail-closed runtime validation and deterministic error precedence.
- Added compatibility checks for model, revision, backend, dimension,
  normalized policy, and 16000 Hz input contract.
- Added privacy, immutability, exception sanitization, and core-only tests.
- Added ADR-005 and Phase 5A technical documentation.

## Assumptions

- Python remains constrained to `>=3.11,<3.12` pending a separate dependency
  compatibility review.
- The MVP remains a modular monolith.
- Current embeddings share the unchanged Phase 3 preprocessing v1 pipeline.
- Raw cosine similarity has no probability or identity-decision meaning.

## Open Questions

- What approved labeled dataset and evaluation protocol can support Phase 5B?
- Which privacy, consent, retention, and legal controls are required before any
  real biometric evaluation?
- How should a future preprocessing contract version be represented before
  pipeline changes are introduced?
- Which robustness, fairness, and anti-spoofing evaluations are required before
  production consideration?

## Known Risks

- Speaker embeddings are sensitive biometric templates.
- Same-model metadata compatibility does not prove biometric accuracy.
- Score distributions can shift with model, preprocessing, channel, language,
  recording device, noise, and population changes.
- Cosine similarity is not probability or confidence without calibration.
- Phase 5A has no threshold, verdict, enrollment, anti-spoofing, or production
  biometric controls.

## Next Step

No implementation phase is authorized. Phase 5B remains unstarted and requires
separate CTO approval after its dataset, consent, privacy, evaluation, and
architecture requirements are reviewed.

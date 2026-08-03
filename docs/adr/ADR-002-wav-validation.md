# ADR-002: WAV Loading And Technical Validation

Status: Accepted and merged in Phase 2.

## Decision

VoiceID Phase 2 implements local WAV technical validation as a modular
monolith component using Python standard library APIs.

The implementation uses:

- a validation policy dataclass for supported technical limits and thresholds;
- typed domain models for metadata, warnings, errors, and status;
- a small RIFF/WAVE header parser to classify container and codec failures;
- standard-library `wave` decoding for PCM16 signal statistics;
- an application service that orchestrates validation without exposing paths.

Phase 2 does not add audio or ML dependencies.

## Alternatives

1. Add `soundfile`, `librosa`, or `torchaudio` now.
2. Implement Phase 2 as a FastAPI or Streamlit endpoint.
3. Convert all input to mono 16000 Hz during validation.
4. Return free-text validation messages without stable issue codes.
5. Treat technical validation as speaker-verification suitability.

## Reason

The approved Phase 2 scope is narrow: accept local RIFF/WAVE PCM16 files,
extract technical metadata, and return structured validation results. The
standard library can safely handle this without introducing heavy dependency
compatibility, license, model-download, or CI runtime risk.

A minimal RIFF/WAVE parser is still useful because the standard `wave` module
does not expose every unsupported codec/container failure with product-stable
semantics. The parser lets VoiceID return controlled error codes while still
using `wave` for actual PCM16 reading.

Keeping the use case framework-neutral preserves the modular monolith selected
in ADR-001 and avoids prematurely adding API or UI layers.

## Consequences

- Phase 2 remains fast, deterministic, and offline.
- No real audio, embeddings, model weights, or ML dependencies are introduced.
- Public validation results contain stable error and warning codes.
- Public responses do not expose absolute or canonical filesystem paths.
- Stereo and non-16 kHz files are accepted only with warnings and are not
  transformed.
- Speaker-verification suitability, scoring, VAD, downmix, resampling, and
  normalization remain deferred to later phases.

## Notes

Technical validation does not guarantee suitability for speaker verification.
Similarity scores and probability-like outputs remain forbidden until later
model, evaluation, threshold, and calibration work.

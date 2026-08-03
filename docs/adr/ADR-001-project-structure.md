# ADR-001: Project Structure And Phase 1 Foundation

Status: Accepted

## Decision

VoiceID Phase 1 uses:

- Python 3.11 with `requires-python = ">=3.11,<3.12"`;
- `src`-layout with the importable package at `src/voiceid`;
- `pyproject.toml` for packaging and tool configuration;
- a modular monolith for the MVP;
- only lightweight development dependencies: pytest, Ruff, and mypy.

## Alternatives

1. Use the draft `app/` layout from the initial specification.
2. Use Python 3.12 or newer immediately.
3. Add future ML, audio, API, and UI dependencies now.
4. Split the MVP into services from the beginning.
5. Keep configuration in ad hoc module-level constants only.

## Reason

The `src`-layout keeps imports honest and makes packaging behavior closer to
real installation. Python 3.11 is a conservative runtime target for the future
ML stack while Phase 1 has no reason to require newer Python features.
The `>=3.11,<3.12` range is temporary and will be revisited before adding
audio and ML dependencies.

`pyproject.toml` gives one standard place for package metadata and tool
configuration. A modular monolith keeps the project simple while preserving
clear boundaries for future audio, embedding, similarity, service, and UI
modules.

Heavy dependencies are intentionally deferred until the phases that need them.
This keeps CI fast, avoids model download side effects, and reduces dependency
compatibility risk while the project foundation is still being established.

## Consequences

- The package can be installed and imported consistently.
- Tooling is simple and reproducible.
- Future phases can add audio and ML modules without changing the project
  foundation.
- The initial specification remains product context, but the Phase 1 layout
  supersedes its draft `app/` tree.
- Python 3.12+ support is not promised until the future ML stack is checked.

## Notes

Future similarity scores must not be presented as match probabilities without
experimental calibration on labeled same-speaker and different-speaker pairs.

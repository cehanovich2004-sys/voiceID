# ADR-006: Experimental Calibration Protocol Boundary

Status: Proposed for PR B1 CTO review.

## Decision

VoiceID will prepare Phase 5B calibration through a documentation-first PR B1
before adding executable experiment contracts or tooling.

PR B1 defines:

- protocol identifier `phase5b-experimental-calibration-protocol-v1`;
- formal experimental protocol;
- privacy and data-governance plan;
- requirements for future experiment contracts;
- approval gates for PR B2 and later calibration tooling.

PR B1 does not add Python modules, dataclasses, runtime APIs, dataset loaders,
metrics implementation, threshold selection, or identity decisions.

The future experiment must use the existing processing provenance from PR A:

- preprocessing contract `phase3-v1`;
- embedding contract `phase4b-v1`;
- backend adapter version `speechbrain-ecapa-adapter-v1`;
- raw cosine comparison version `1`.

## Alternatives

1. Add executable experiment contracts immediately in PR B.
2. Build a calibration runner and metrics code together with the protocol.
3. Keep Phase 5B only as roadmap text until a dataset exists.
4. Select an initial threshold before a formal experiment plan.

## Reason

Documentation-first sequencing keeps the biometric boundary explicit before
the project touches real data or creates experiment execution code. It also
avoids approving a public Python API before the dataset, privacy, reporting,
and partition semantics are fully reviewed.

Splitting PR B allows CTO review of the research protocol and governance plan
without coupling it to executable contracts. Future PR B2 can then propose the
minimal API shape with clearer requirements and fewer speculative abstractions.

## Consequences

- Phase 5B is not complete after PR B1.
- No calibration is executed and no threshold is selected.
- No public runtime API changes are introduced by PR B1.
- Future PR B2 requires separate CTO approval for exact contract names,
  fields, validation behavior, and package exports.
- Future calibration tooling requires another approval after contracts exist.
- Version equality remains reproducibility metadata, not proof of biometric
  accuracy or legal compliance.
- Real data remains blocked until privacy/legal and dataset approval.

## Future Contract Direction

Future contracts should be immutable, privacy-safe, and fail closed. They may
need to represent protocol identity, approved sample manifest metadata,
comparison pair metadata, partition roles, comparison class, provenance, and
reporting policy.

The exact Python module, class names, fields, and serialization shape are not
approved by this ADR. They remain PR B2 design questions.

## Non-Goals

PR B1 does not implement:

- executable experiment contracts;
- dataset acquisition or ingestion;
- pair generation;
- FAR, FRR, EER, ROC, DET, or confidence-interval computation;
- threshold search or runtime threshold API;
- `MATCH`, `NO_MATCH`, `UNCERTAIN`, accept/reject, or identity verdicts;
- enrollment storage;
- processing algorithm changes;
- new dependencies;
- real or synthetic biometric artifacts.

## Risks

- A documentation-only protocol can become stale if PR B2 changes contract
  semantics. PR B2 must update this ADR or add a superseding ADR if needed.
- Dataset restrictions may force changes to partitioning or reporting plans.
- Privacy/legal review may prohibit some planned strata or reporting outputs.
- Calibration metrics can be misread as production readiness; documentation
  must keep the non-decision boundary visible.

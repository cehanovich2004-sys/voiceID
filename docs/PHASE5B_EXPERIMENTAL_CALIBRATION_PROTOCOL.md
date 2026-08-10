# Phase 5B Experimental Calibration Protocol

Status: protocol/privacy prerequisite documented. Calibration implementation
and execution have not started. PR B2 adds minimal executable plan contracts
for future CTO-reviewed calibration work; it does not execute calibration.

Protocol identifier: `phase5b-experimental-calibration-protocol-v1`.

This document defines the controlled experiment that a future Phase 5B
implementation must follow before VoiceID can discuss biometric thresholds or
decision policy. It does not implement dataset ingestion, metrics computation,
threshold selection, reporting code, or runtime identity decisions.

## Implemented Now

- Formal experimental calibration protocol.
- Privacy and data-governance requirements for future experiment work.
- Requirements for future experiment contracts and validation boundaries.
- Architecture decision record for the experiment boundary.
- Minimal `voiceid.calibration` contracts for protocol identity, processing
  provenance, sample records, pair records, experiment plans, and trusted plan
  validation.

## Not Implemented

- Dataset acquisition, loader, ingestion pipeline, or manifest parser.
- Calibration runner.
- FAR, FRR, EER, ROC, DET, or confidence-interval computation.
- Threshold search, threshold selection, or runtime threshold API.
- `MATCH`, `NO_MATCH`, `UNCERTAIN`, accept/reject, or identity verdicts.
- Enrollment storage or production biometric decision logic.

## Future Approval Gates

1. CTO approval of PR B1 protocol and privacy plan.
2. CTO approval of PR B2 public experiment-plan contracts.
3. Privacy/legal and dataset approval before any real data is obtained or used.
4. Separate approval for calibration tooling and experiment execution.
5. Results review before any operating-point or threshold decision.

## Research Question

For a fixed VoiceID processing stack, how do raw cosine similarity scores
between compatible speaker embeddings distribute for same-speaker and
different-speaker comparison pairs under approved recording conditions?

The experiment may describe threshold-dependent behavior, but it must not
select a production threshold without a separate product, security, privacy,
and CTO decision.

## Unit Of Observation

The unit of observation is one comparison pair:

- a reference-side embedding produced from one approved audio sample; and
- a probe-side embedding produced from another approved audio sample; and
- one raw cosine similarity score computed by the Phase 5A comparison
  contract.

The pair is labeled before score analysis as either `genuine` or `impostor`.
Labels must be derived from approved, auditable dataset metadata, not inferred
from filenames, storage paths, or model output.

## Comparison Classes

`genuine` comparison:

- reference and probe samples belong to the same approved pseudonymous subject;
- reference and probe must be different recording samples;
- the same source audio fragment, duplicate file, trimmed derivative, codec
  derivative, or otherwise derived copy must not appear on both sides.

`impostor` comparison:

- reference and probe samples belong to different approved pseudonymous
  subjects;
- subject identity must be established by approved dataset metadata;
- demographic, condition, or session metadata must not be used as a substitute
  for identity labels.

The experiment must report how repeated samples and correlated comparisons are
handled. Correlated pair generation can inflate apparent certainty if many
pairs share the same source sample or subject.

## Reference And Probe Separation

The protocol uses explicit reference/enrollment-side and probe-side roles only
for experiment design. This does not introduce enrollment storage in the
product.

Rules:

- A source audio fragment or derivative must not appear on both sides of one
  comparison.
- The reference partition for threshold exploration must not leak into the
  independent test partition.
- Any sample reuse across comparisons must be documented and considered when
  reporting uncertainty.
- Speaker-disjoint splits are required wherever the experiment estimates
  generalization to unseen speakers.

## Partitioning And Leakage Prevention

The experiment must define partitions before score analysis:

- development/calibration partition;
- independent test partition;
- optional holdout or audit partition if required by the dataset plan.

Thresholds, operating points, and calibration alternatives may only be explored
on the development/calibration partition. The test partition is used once for
independent evaluation of already fixed decisions.

Forbidden leakage:

- using test results to adjust thresholds;
- mixing source fragments or derivatives across partitions;
- choosing exclusion criteria after seeing score distributions;
- using the same speaker/session constraints inconsistently between partitions;
- tuning preprocessing, embedding, or comparison behavior on the test
  partition.

## Stratification And Reporting

The experiment plan must define approved reporting strata before execution.
Relevant strata may include recording device, channel, source sample rate,
duration band, language or accent metadata when approved, session gap, noise
condition, and collection source.

Stratification must follow data-minimization and privacy approvals. Sensitive
attributes must not be collected or reported unless there is a documented
purpose, lawful basis, consent or equivalent approval, and a reporting plan
that prevents re-identification.

Aggregate metrics do not replace subgroup or condition analysis. The final
report must state when a condition has insufficient data for a meaningful
estimate.

## Sample Adequacy

Minimum sample adequacy is a required planning item. This protocol does not
invent a universal number of speakers, samples, or pairs.

The experiment plan must justify the sample size for its intended research
question, expected uncertainty, recording conditions, and reporting strata.
Any underpowered result must be labeled as exploratory.

## Handling Invalid Or Incompatible Results

Phase 2, Phase 3, Phase 4B, and Phase 5A can produce controlled `INVALID`
results. The experiment must define how these are counted before execution.

Required behavior:

- invalid preprocessing, embedding, or similarity results are not silently
  converted into scores;
- invalid and incompatible counts are reported by stable code;
- no partial waveform, embedding, path, or raw exception is stored in reports;
- exclusion from scoring must follow pre-defined criteria;
- invalid handling must not be used to tune a threshold after seeing results.

## Metrics Semantics

Future documentation or tooling may define these metrics, but PR B1 does not
implement them.

False acceptance rate:

```text
FAR(threshold) =
    false accepted impostor comparisons / all evaluated impostor comparisons
```

False rejection rate:

```text
FRR(threshold) =
    false rejected genuine comparisons / all evaluated genuine comparisons
```

Threshold-dependent trade-off:

- raising a threshold can reduce false acceptance while increasing false
  rejection;
- lowering a threshold can reduce false rejection while increasing false
  acceptance;
- the acceptable operating point depends on threat model, product UX,
  population, channel, recording conditions, and risk tolerance.

EER:

- equal error rate is a descriptive experimental summary where FAR and FRR are
  approximately equal;
- EER is not automatically a production threshold.

ROC and DET:

- ROC and DET curves are analysis artifacts for threshold-dependent behavior;
- they do not by themselves approve a biometric decision policy.

Uncertainty:

- confidence intervals or another approved uncertainty estimate are mandatory
  reporting items;
- repeated samples and correlated comparisons must be accounted for when
  interpreting uncertainty.

Raw cosine similarity is not probability, confidence, or identity evidence by
itself. A biometric performance experiment does not prove legal compliance,
production safety, fairness, or anti-spoofing capability.

## Version And Provenance Requirements

Every evaluated score must be tied to the processing versions already merged
through PR A:

- preprocessing contract: `phase3-v1`;
- embedding contract: `phase4b-v1`;
- backend adapter: `speechbrain-ecapa-adapter-v1`;
- raw cosine comparison: `1`.

The experiment must also record:

- repository commit SHA;
- model identifier and pinned model revision;
- optional dependency versions used for embedding extraction;
- Python and platform version;
- dataset manifest identifier or approved audit reference;
- partition assignment provenance;
- protocol identifier `phase5b-experimental-calibration-protocol-v1`.

Version equality supports reproducibility and compatibility. It does not prove
biometric accuracy, calibration quality, production readiness, or legal
compliance.

## Reproducibility Requirements

A future experiment run must be reproducible from approved non-sensitive
metadata and controlled external storage.

Required records:

- exact code commit;
- exact protocol document version;
- approved dataset manifest reference;
- partition generation method;
- pair generation method;
- exclusion criteria;
- processing provenance from Phase 3, Phase 4B, and Phase 5A;
- dependency versions;
- random seed if randomized procedures are approved.

Secrets, absolute storage paths, raw audio, derived waveforms, embeddings, and
speaker identities must not be written to public metadata, Git history, CI
artifacts, user-facing errors, or normal logs.

## Predefined Exclusion Criteria

The experiment plan must define exclusion criteria before execution. Examples
that may be appropriate after approval:

- unsupported or invalid WAV input;
- invalid Phase 3 preprocessing result;
- invalid Phase 4B embedding result;
- incompatible Phase 5A comparison result;
- missing approved consent or legal basis;
- manifest record with inconsistent pseudonymous subject metadata;
- sample that violates partition, source-fragment, or derivative restrictions.

Exclusions must be counted and reported. They must not be selected after
looking at score distributions.

## Calibration Contract Boundary

PR B2 introduces the approved minimal contract namespace
`voiceid.calibration`.

Public constants:

- `CALIBRATION_PROTOCOL_IDENTIFIER`;
- `CALIBRATION_CONTRACT_VERSION`.

Public roles:

- `CalibrationPartition`;
- `CalibrationComparisonClass`;
- `CalibrationProtocolIdentity`;
- `CalibrationProcessingProvenance`;
- `CalibrationSampleRecord`;
- `CalibrationPairRecord`;
- `CalibrationExperimentPlan`;
- `validate_calibration_experiment_plan`.

These contracts are declarative. They do not load datasets, generate pairs,
read files, compute scores, calculate metrics, select thresholds, serialize
audit manifests, or return identity decisions.

Contract validation follows the existing VoiceID style:

Future executable contracts must follow the existing VoiceID style:

- immutable objects;
- exact types with no implicit coercion;
- `bool` must not be accepted where an `int` is required;
- required strings must be exact `str`, non-empty, and not whitespace-only;
- `bytes`, `None`, and arbitrary objects with `__str__` must be rejected;
- enum-like values must be closed sets;
- nested collections must be deeply validated and copied from caller-owned
  inputs;
- forged state must be rechecked at validation/serialization boundaries;
- invalid state must fail closed with stable safe errors;
- error text, `repr()`, `str()`, and public serialization must not echo
  sensitive malformed payloads.

Protocol identity record:

- Need: tie results to the approved protocol.
- Allowed data: protocol identifier, protocol document version, repository
  commit SHA.
- Forbidden data: filesystem paths, hostnames, speaker identifiers, secrets.
- Boundary: constructed by trusted experiment orchestration, not raw caller
  input.
- B2 shape: `CalibrationProtocolIdentity`.

Sample manifest metadata:

- Need: describe approved samples without storing raw audio or speaker names.
- Allowed data: pseudonymous experiment-local subject ID, sample ID, approved
  partition, approved condition labels, consent/audit reference.
- Forbidden data: raw audio bytes, waveform, embedding, filesystem path, real
  name, phone number, account ID, token, or speaker identifier outside the
  experiment pseudonym namespace.
- Boundary: validated from an approved manifest; caller-owned collections must
  be copied.
- B2 shape: `CalibrationSampleRecord` with pseudonymous sample, subject, source
  group, and partition fields only.

Comparison pair metadata:

- Need: represent genuine/impostor pair labels and reference/probe roles.
- Allowed data: pseudonymous sample IDs, comparison class, partition role,
  approved provenance references.
- Forbidden data: score threshold, identity verdict, raw score if the contract
  is only a pair-plan record, audio, waveform, embedding, paths, speaker names.
- Boundary: validated before score computation; forged contradictory pair state
  must fail closed.
- B2 shape: `CalibrationPairRecord` with pair id, reference sample id, probe
  sample id, comparison class, and partition.

Out of scope for PR B2:

- reporting-policy contracts;
- consent/audit-reference fields;
- condition labels and strata;
- correlation-group fields;
- score records.

## Remaining Open API Decisions

The following remain unresolved after the minimal B2 contract layer:

- Whether score-level records belong in calibration tooling or a later
  contracts PR.
- Whether audit serialization should exist and which fields it may expose.
- How repeated/correlated-comparison grouping should be represented.
- Which approved condition labels are mandatory after dataset and privacy
  review.
- Exact reporting-policy contract shape.

## Non-Decision Statement

This protocol prepares future experimental work. It does not authorize data
collection, experiment execution, production biometric verification, threshold
selection, probability output, or identity decisions.

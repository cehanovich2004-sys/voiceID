# Phase 5B Experimental Calibration Protocol

Status: PR B1 documentation prerequisite. Calibration implementation and
execution have not started.

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

## Not Implemented

- Executable experiment contracts.
- Dataset acquisition, loader, ingestion pipeline, or manifest parser.
- Calibration runner.
- FAR, FRR, EER, ROC, DET, or confidence-interval computation.
- Threshold search, threshold selection, or runtime threshold API.
- `MATCH`, `NO_MATCH`, `UNCERTAIN`, accept/reject, or identity verdicts.
- Enrollment storage or production biometric decision logic.

## Future Approval Gates

1. CTO approval of PR B1 protocol and privacy plan.
2. Privacy/legal and dataset approval before any real data is obtained or used.
3. Exact API proposal for PR B2 experiment contracts.
4. CTO approval of the public contract shape.
5. Separate implementation PR B2 for executable contracts.
6. Separate approval for calibration tooling and experiment execution.
7. Results review before any operating-point or threshold decision.

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

## Future Experiment Contracts Requirements

PR B2 may introduce executable contracts only after CTO approval of the exact
API shape. The names below describe responsibilities, not approved Python
classes or modules.

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
- Open questions: whether this should be a standalone object or a field on an
  experiment plan.

Sample manifest metadata:

- Need: describe approved samples without storing raw audio or speaker names.
- Allowed data: pseudonymous experiment-local subject ID, sample ID, approved
  partition, approved condition labels, consent/audit reference.
- Forbidden data: raw audio bytes, waveform, embedding, filesystem path, real
  name, phone number, account ID, token, or speaker identifier outside the
  experiment pseudonym namespace.
- Boundary: validated from an approved manifest; caller-owned collections must
  be copied.
- Open questions: exact required condition labels and manifest format require
  privacy/legal and dataset approval.

Comparison pair metadata:

- Need: represent genuine/impostor pair labels and reference/probe roles.
- Allowed data: pseudonymous sample IDs, comparison class, partition role,
  approved provenance references.
- Forbidden data: score threshold, identity verdict, raw score if the contract
  is only a pair-plan record, audio, waveform, embedding, paths, speaker names.
- Boundary: validated before score computation; forged contradictory pair state
  must fail closed.
- Open questions: whether repeated/correlated-pair grouping should be explicit
  in the pair contract or in reporting metadata.

Reporting policy metadata:

- Need: preserve pre-declared reporting strata, exclusion handling, and
  uncertainty requirements.
- Allowed data: approved condition labels, aggregation policy, uncertainty
  method name after approval.
- Forbidden data: per-speaker public output by default, secrets, raw biometric
  data, threshold values unless a later PR explicitly approves threshold
  analysis artifacts.
- Boundary: validated before experiment execution.
- Open questions: exact uncertainty method and subgroup reporting rules require
  experiment design approval.

## Future PR B2 Open API Decisions

The following are intentionally unresolved until PR B2:

- Python module name and package export location, if any.
- Whether protocol identity is a standalone contract or part of an experiment
  plan contract.
- Exact field names for pseudonymous subject/sample identifiers.
- Whether partition roles and comparison classes are separate objects or enum
  fields.
- Whether score-level records are in scope for PR B2 or should wait for
  calibration tooling.
- Serialization shape and whether `to_dict()` is part of the public contract.
- Stable error-code set for experiment-plan validation.
- How repeated/correlated-comparison grouping should be represented.
- Which approved condition labels are mandatory after dataset and privacy
  review.

## Non-Decision Statement

This protocol prepares future experimental work. It does not authorize data
collection, experiment execution, production biometric verification, threshold
selection, probability output, or identity decisions.

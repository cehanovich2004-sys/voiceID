# Phase 5B Privacy And Data Governance Plan

Status: PR B1 documentation prerequisite. No real biometric data is approved,
collected, ingested, stored, or processed by this document.

This plan defines privacy-first requirements for any future Phase 5B
experimental calibration work. It is not a legal compliance claim. VoiceID must
obtain separate privacy/legal review before using real audio, waveforms,
embeddings, subject metadata, or score reports.

## Approval Checkpoint

Before any real biometric data is obtained or used, the project needs explicit
manual approval for:

- dataset source and terms;
- documented consent or another separately confirmed lawful basis;
- purpose limitation;
- storage location and access-control boundary;
- retention and deletion plan;
- incident handling;
- export/reporting policy;
- experiment protocol and version.

No developer may use real voice recordings, derived waveforms, embeddings, or
speaker metadata in Git, local fixtures, CI, logs, or ad hoc experiments before
this approval.

## Purpose Limitation

Allowed future purpose:

- evaluate raw cosine score distributions for one-to-one speaker verification
  research under the approved Phase 5B protocol.

Not allowed without separate approval:

- production identity decisions;
- enrollment database creation;
- speaker identification;
- anti-spoofing claims;
- training or fine-tuning models;
- reusing data for unrelated products or demos.

## Data Minimization

Future experiment records should contain only what is needed for the approved
research question, reproducibility, and audit.

Preferred identifiers:

- pseudonymous experiment-local subject IDs;
- pseudonymous sample IDs;
- approved partition labels;
- approved condition labels.

Forbidden in repository, public metadata, errors, logs, and reports:

- raw audio bytes;
- derived waveform values;
- speaker embedding values;
- real speaker names;
- phone numbers, account IDs, or external customer IDs;
- absolute filesystem paths;
- cache paths;
- access tokens or secrets;
- speaker identifiers in filenames;
- raw exception text containing sensitive values.

## Storage Boundary

Real data and sensitive derived artifacts must be stored outside the Git
repository in a controlled storage boundary approved before use.

Git must not contain:

- real recordings;
- synthetic records presented as real experiment results;
- embeddings;
- model cache files or weights;
- dataset manifests with direct identifiers;
- CI artifacts containing biometric data;
- local absolute paths or secrets.

Raw audio, derived waveforms, embeddings, and aggregate reports should be
separated by storage class. Access to each class should be granted only to
people who need it for the approved experiment.

## Access Control

Future experiment work must define:

- who can access raw audio;
- who can access embeddings;
- who can access score-level outputs;
- who can publish aggregate reports;
- how access is granted, reviewed, and revoked.

Default posture: fewer people should access raw audio than aggregate reports.
Embeddings are sensitive biometric templates and should be protected at least
as carefully as raw audio.

## Provenance And Audit

An approved experiment must preserve an audit record without exposing sensitive
values.

Required audit fields should include:

- approved dataset reference;
- consent or legal-basis reference;
- pseudonymous subject/sample IDs;
- partition assignment;
- protocol identifier;
- repository commit SHA;
- processing contract versions;
- model identifier and pinned revision;
- exclusion counts and stable invalid-result codes.

Audit records must not include raw audio, waveform values, embedding values,
speaker names, tokens, secrets, or absolute storage paths.

## Retention And Deletion

Before using data, the project must define:

- retention period for raw audio;
- retention period for embeddings;
- retention period for score-level outputs;
- deletion procedure;
- deletion verification record;
- conditions for early deletion.

Retention must be tied to the approved research purpose. Data must not be kept
indefinitely by default.

## Reports And Exports

Reports should be aggregate by default.

Allowed after approval:

- aggregate score distributions;
- aggregate FAR/FRR/EER summaries after implementation approval;
- stratified aggregate metrics when sample adequacy and privacy review allow
  reporting.

Forbidden by default:

- per-speaker public results;
- filenames or IDs that reveal a person;
- raw audio, waveform, or embedding excerpts;
- direct identifiers;
- sensitive small-cell subgroup reports that risk re-identification;
- threshold or production decision recommendations unless separately approved.

## Logs And Errors

Logs and user-facing errors must stay privacy-safe:

- stable error codes are allowed;
- controlled messages are allowed;
- raw exception text is not allowed;
- paths, tokens, filenames with speaker identifiers, waveforms, embeddings, and
  per-speaker labels are not allowed.

If an incident may have exposed sensitive data, development should stop and the
incident handling process must be followed before continuing.

## Incident Handling

The future experiment plan must include:

- how to report suspected data exposure;
- who triages the incident;
- how affected artifacts are identified;
- how secrets are rotated if needed;
- how data is deleted or quarantined;
- how the project records remediation.

This plan does not claim compliance with GDPR, BIPA, CCPA, or any other law.
Legal/privacy review must decide which obligations apply before real biometric
processing begins.

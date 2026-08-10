# Phase 5B Feasibility Probe

Status: proposed implementation for an opt-in local feasibility check. This is
not production biometric verification and does not select an operating
threshold.

The probe answers a narrow engineering question:

```text
Can the current local VoiceID stack produce visibly separated exploratory raw
cosine score distributions for a small approved WAV manifest?
```

It runs:

```text
local JSON manifest
-> Phase 3 WAV preprocessing
-> Phase 4B speaker embeddings from a prepared local cache
-> Phase 5A raw cosine similarity
-> CSV and static HTML exploratory report
```

## Non-Goals

- No dataset acquisition or ingestion pipeline.
- No network access or model download during the experiment.
- No embeddings, WAV files, manifests, or personal data in Git.
- No enrollment storage.
- No API, UI, database, anti-spoofing, diarization, VAD, or identification.
- No FAR/FRR/EER production implementation beyond exploratory report metrics.
- No threshold selection, recommendation, probability, confidence, `MATCH`, or
  `NO_MATCH`.

## Manifest Grammar

The input manifest is local JSON and must not be committed if it contains real
paths or biometric data references.

Top-level object:

```json
{
  "repository_commit_sha": "<40 hex characters>",
  "samples": [],
  "thresholds": [-0.5, 0.0, 0.5]
}
```

`thresholds` is optional. If omitted, the probe uses an exploratory grid from
`-1.0` to `1.0` in `0.05` steps.

Each sample record:

```json
{
  "sample_id": "smp_0123456789abcdef0123456789abcdef",
  "subject_id": "sub_0123456789abcdef0123456789abcdef",
  "source_group_id": "src_0123456789abcdef0123456789abcdef",
  "partition": "CALIBRATION",
  "wav_path": "/local/not-in-git/audio.wav"
}
```

Identifier grammar follows the Phase 5B calibration contracts:

- `sample_id = smp_<32 lowercase hex>`;
- `subject_id = sub_<32 lowercase hex>`;
- `source_group_id = src_<32 lowercase hex>`;
- generated internal `pair_id = pair_<32 lowercase hex>`.

`wav_path` exists only in the manifest adapter and is never added to
calibration contracts, result objects, CSV reports, HTML reports, logs, or
public errors.

## Pair Generation

Pairs are generated deterministically from manifest samples:

- samples are sorted by partition, subject, source group, and sample ID;
- only samples in the same partition are compared;
- reference and probe must be different samples;
- reference and probe must have different `source_group_id`;
- same subject creates a `GENUINE` pair;
- different subjects create an `IMPOSTOR` pair.

Pair IDs are deterministic internal opaque IDs. They are not written to reports.

`CALIBRATION` is the only partition used for exploratory thresholds, FAR/FRR,
score overlap, and the final feasibility label. `HOLDOUT` scores may be
generated and listed in `scores.csv` for later inspection, but they do not
influence exploratory metrics, label criteria, or any threshold-related
summary. There is no fallback from `HOLDOUT` to `CALIBRATION`.

## Reports

The output directory contains:

- `scores.csv`: row index, partition, comparison class, raw cosine score for
  evaluated pairs from any partition;
- `threshold_metrics.csv`: `CALIBRATION`-only exploratory threshold, FAR, FRR,
  and counts;
- `summary.csv`: aggregate counts, overlap, invalid counts, final label;
- `report.html`: static local report with summary, distributions, histogram,
  and exploratory FAR/FRR table.

Reports must not contain:

- WAV paths or filenames;
- sample, subject, source-group, or pair IDs;
- waveforms or embedding values;
- tokens, secrets, cache paths, or raw exception text.

Issue codes are accepted only from trusted Phase 3, Phase 4B, and Phase 5A
contracts. Unknown, malformed, or forged codes are collapsed to
`<stage>.unknown`. Raw issue strings, paths, tokens, sample identifiers,
waveform-like values, and exception messages are not serialized to CSV, HTML,
stdout, stderr, or public exceptions.

## Metrics

FAR and FRR are reported only for `CALIBRATION` exploratory thresholds:

```text
FAR(threshold) = impostor scores >= threshold / evaluated impostor scores
FRR(threshold) = genuine scores < threshold / evaluated genuine scores
```

Raw cosine similarity is a score, not probability, confidence, or a biometric
identity verdict. Exploratory thresholds in the report are not production
thresholds.

## Label Criteria

Label criteria version: `phase5b-feasibility-label-v1`.

Constants:

- minimum evaluated genuine scores: `2`;
- minimum evaluated impostor scores: `2`;
- `PROMISING` maximum exploratory FAR and FRR: `0.10`;
- `NOT PROMISING` minimum best balanced error: `0.35`.

Label rules:

- `INCONCLUSIVE` if evaluated genuine or impostor scores are below the minimum,
  or no threshold metrics are available;
- `PROMISING` if score ranges do not overlap and at least one exploratory
  threshold has both FAR and FRR less than or equal to `0.10`;
- `NOT PROMISING` if the best balanced error, `min(max(FAR, FRR))`, is greater
  than or equal to `0.35`;
- otherwise `INCONCLUSIVE`.

These labels are engineering feasibility summaries only. They are not
biometric accept/reject decisions.

## Offline Model Boundary

Production use of the probe constructs the SpeechBrain ECAPA backend with
`offline=True`. The model must already be present in the local cache. The probe
does not call the bootstrap/download path and does not perform network access.

Unit tests use a fake embedding service and synthetic temporary WAV files.

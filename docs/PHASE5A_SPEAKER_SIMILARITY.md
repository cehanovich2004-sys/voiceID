# Phase 5A: Safe Speaker Embedding Similarity

Status: implemented in a feature branch, pending independent QA, CTO review,
and merge.

Phase 5A compares two compatible `SpeakerEmbeddingResult` objects with raw
cosine similarity. It is a deterministic, core-only calculation: it does not
load a model, access the network, read audio, or persist embeddings.

## Public API

```python
from voiceid.services import compare_speaker_embeddings

result = compare_speaker_embeddings(reference, candidate)
print(result.to_dict())
```

The function accepts only two Phase 4B embedding results and returns a
`SpeakerSimilarityResult` with status `VALID` or `INVALID`.

`VALID` contains:

- `similarity`: a finite Python `float` in `[-1.0, 1.0]`;
- safe comparison metadata;
- no errors.

`INVALID` contains:

- no similarity value;
- no metadata;
- exactly one stable, privacy-safe error.

## Metric And Numeric Policy

For reference embedding `a` and candidate embedding `b`:

```text
cosine_similarity = dot(a, b) / (L2(a) * L2(b))
```

Input arrays remain `float32`. Dot product and L2 norms use `float64`
accumulation. The returned value is a Python `float`. A final clip to
`[-1.0, 1.0]` guards only against floating-point overshoot; it is not score
calibration or normalization. Normalized vector copies are not retained.

An embedding with L2 norm less than or equal to `1e-8` is rejected as
`ZERO_OR_NEAR_ZERO_EMBEDDING`. This is a numerical degeneracy guard, not a
speaker-verification decision threshold.

## Validation And Compatibility

Both inputs are checked again at runtime even though Phase 4B contracts enforce
their own invariants. Each input must be a `VALID` `SpeakerEmbeddingResult`
with:

- a one-dimensional NumPy array;
- exact `float32` dtype;
- shape `(192,)` matching metadata;
- finite values;
- finite L2 norm greater than `1e-8`;
- input sample rate `16000` Hz.

The two results must match on:

- embedding dimension;
- model identifier;
- pinned model revision;
- backend name;
- raw/normalized policy.

Inference or recording device metadata is not compared. No automatic dtype,
dimension, backend, revision, or normalization conversion is performed.

## Error Precedence

Phase 5A returns one error in this stable order:

1. invalid reference object or status;
2. invalid candidate object or status;
3. invalid reference embedding;
4. invalid candidate embedding;
5. zero or near-zero reference;
6. zero or near-zero candidate;
7. incompatible metadata;
8. comparison failure.

Stable codes are:

- `INVALID_REFERENCE`;
- `INVALID_CANDIDATE`;
- `INVALID_EMBEDDING`;
- `ZERO_OR_NEAR_ZERO_EMBEDDING`;
- `INCOMPATIBLE_EMBEDDINGS`;
- `COMPARISON_ERROR`.

Unexpected ordinary exceptions and `MemoryError` become a sanitized
`COMPARISON_ERROR`. `KeyboardInterrupt` and `SystemExit` are not intercepted.

## Privacy

Embeddings remain sensitive biometric templates. Phase 5A does not expose or
store embedding values, previews, vector norms, waveforms, file paths, cache
paths, access tokens, or raw exception details. `repr()`, `str()`, `to_dict()`,
errors, and normal execution contain only the raw score and safe metadata.
Inputs are not modified and remain read-only.

## Preprocessing Compatibility

Current Phase 4B embeddings are considered compatible only within the existing
Phase 3 preprocessing v1 pipeline. Phase 5A relies on that unchanged pipeline
and exact agreement of currently available embedding metadata.

`EmbeddingMetadata` does not gain a preprocessing version in this phase. Any
future preprocessing change requires a separate ADR and an explicit version
field before results from different pipelines can be mixed.

## Interpretation And Non-Goals

Cosine similarity is a raw score. It is not probability, confidence, or proof
of identity. Phase 5A does not choose a biometric decision threshold and does
not return `MATCH`, `NO_MATCH`, or `UNCERTAIN`.

Phase 5B is a separate experimental calibration phase requiring labeled data,
evaluation methodology, and explicit CTO approval. Production biometric
verification is not ready after Phase 5A.

Phase 5A adds no dependencies and works with the core installation. It does not
perform model inference, model download, enrollment, identification, API/UI/DB
work, or anti-spoofing.

# ADR-005: Raw Cosine Similarity Boundary

Status: Accepted for Phase 5A implementation, pending independent QA and merge.

## Decision

VoiceID Phase 5A compares two compatible Phase 4B speaker embeddings with a
pure deterministic cosine similarity function.

- Inputs are raw `float32` embeddings with exact shape `(192,)`.
- Dot product and L2 norms use `float64` accumulation.
- L2 norm less than or equal to `1e-8` is rejected as numerically degenerate.
- The final score is clipped to `[-1.0, 1.0]` only for floating-point overshoot.
- Compatibility requires the same dimension, model identifier, pinned model
  revision, backend, normalized policy, and 16000 Hz input contract.
- Model identifier, revision, and backend remain internal compatibility inputs
  and are omitted from public similarity metadata.
- The result exposes a raw score and safe metadata, never a biometric verdict.

The public comparison function is independent of embedding backends, model
loaders, audio paths, and web frameworks. It works in the core installation
without the optional `embeddings` extra.

## Alternatives

1. Dot product without norm adjustment.
2. Euclidean distance.
3. A third-party metrics dependency such as scikit-learn.
4. Backend-specific verification APIs.
5. Combining similarity with a decision threshold in the same phase.

## Reason

Cosine similarity is simple, deterministic, and appropriate for comparing
fixed-size speaker embedding vectors while remaining independent of vector
magnitude. NumPy already provides the required operations, so another runtime
dependency is unnecessary.

Keeping calculation separate from decision policy prevents an uncalibrated
score from being presented as probability, confidence, or identity evidence.
It also permits later calibration work to evaluate the metric without changing
the Phase 4B embedding backend or Phase 5A contract.

## Consequences

- Phase 5A returns only `VALID` or `INVALID`, never `MATCH`, `NO_MATCH`, or
  `UNCERTAIN`.
- No biometric decision threshold exists in Phase 5A.
- Phase 5B must use labeled evaluation data and a separate ADR before adding
  calibration or verdict policy.
- Runtime validation is fail-closed and returns exactly one stable safe error.
- Writable vectors or inconsistent Phase 4B sample-count/duration metadata are
  rejected without copying or repairing the input.
- `MemoryError` and unexpected ordinary exceptions are sanitized as
  `COMPARISON_ERROR`; process-control exceptions pass through.
- Embeddings and vector norms are not serialized, logged, or persisted.
- Untrusted model metadata strings are not copied to public similarity result
  surfaces.
- Recording/inference device differences do not alone make embeddings
  incompatible.

## Preprocessing Contract

Phase 5A assumes all current Phase 4B embeddings were produced from the
unchanged Phase 3 preprocessing v1 pipeline. No preprocessing version field is
added retroactively. Before a future preprocessing change can coexist with the
current pipeline, a separate ADR and explicit version metadata are required.

## Risks

- Raw cosine similarity has no calibrated biometric meaning by itself.
- Model or preprocessing changes can shift score distributions even when the
  mathematical metric is unchanged.
- Same-model metadata compatibility does not establish real-world biometric
  accuracy, robustness, fairness, or anti-spoofing capability.

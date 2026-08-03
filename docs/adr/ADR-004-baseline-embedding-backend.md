# ADR-004: Baseline Speaker Embedding Backend

Status: Accepted for Phase 4B implementation, pending CTO review of PR.

## Decision

VoiceID Phase 4B uses SpeechBrain ECAPA-TDNN as the first baseline speaker
embedding backend:

- model identifier: `speechbrain/spkrec-ecapa-voxceleb`;
- pinned revision: `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`;
- expected embedding dimension: `192`;
- CPU inference;
- tensor-only extraction with `normalize=False`.

The implementation adds a backend protocol, model loader, immutable embedding
policy, typed result contract, and application service. The service consumes
only Phase 3 preprocessed waveform results and does not know SpeechBrain
internal tensor shapes.

SpeechBrain and PyTorch are optional dependencies under the `embeddings` extra.
The base VoiceID installation remains lightweight.

## Alternatives

1. NVIDIA NeMo TitaNet-Large.
2. WeSpeaker ResNet34.
3. `pyannote/embedding`.
4. A project-specific embedding model.
5. Direct SpeechBrain calls from the application service.

## Reason

SpeechBrain ECAPA-TDNN is the smallest practical baseline for the MVP:

- it directly targets speaker verification and embedding extraction;
- the model repository is ungated;
- the model card and hyperparameters expose a 16 kHz single-channel input
  expectation and 192-dimensional embedding layer;
- it aligns with the Phase 3 output contract;
- dependency spike passed on Python 3.11 with NumPy 2.x;
- local offline inference works after explicit cache preparation.

NeMo TitaNet-Large remains a strong benchmark candidate, but its toolkit stack
is heavier and official examples are file-path oriented. WeSpeaker remains a
good challenger, but the public package guidance and requirements are less
aligned with Python 3.11 and NumPy 2.x. pyannote embedding is gated and brings
access-token and telemetry considerations that are not ideal for the first
baseline.

Keeping the backend behind a protocol lets tests use fake backends and keeps
future model swaps out of the application service contract.

## Consequences

- Runtime model inference requires the optional `embeddings` extra.
- Model cache preparation is explicit and separate from strict offline runtime.
- The expected local snapshot contains only model files, not example audio.
- Missing or incomplete cache produces controlled stable error codes.
- Embedding values are excluded from public dictionary, string, repr, and log
  surfaces.
- Embeddings are treated as sensitive biometric templates.
- Phase 4B returns embeddings only; similarity, probability, thresholds, and
  verdicts remain deferred.
- The zero or near-zero RMS guard rejects degenerate Phase 3 waveforms before
  model inference. This is not VAD.

## Dependency Spike Result

Candidate pins passed on macOS arm64 with Python 3.11.15:

- `speechbrain==1.1.0`;
- `torch==2.8.0`;
- `torchaudio==2.8.0`;
- `huggingface_hub==0.36.2`;
- NumPy `2.4.6`;
- SciPy `1.17.1`.

SpeechBrain returned raw shape `[1, 1, 192]`; the adapter flattens it to public
shape `[192]` and `dtype=float32`. Repeated CPU inference on the same synthetic
waveform was deterministic with max absolute difference `0.0`. Offline reload
from a prepared local snapshot also matched at `atol=1e-6`.

## Risks

- `torchaudio` emits a deprecation warning in 2.8 and states some APIs are
  moving toward maintenance/removal in later releases. Pin drift beyond 2.8
  must be tested before upgrade.
- Model weights and training data need legal review before commercial biometric
  use.
- Full checksum verification for every model file is a production hardening
  item. Phase 4B validates the expected manifest but does not pin per-file
  cryptographic hashes.

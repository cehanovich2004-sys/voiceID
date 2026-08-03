# ADR-003: Deterministic Audio Preprocessing

Status: Accepted and merged.

## Decision

VoiceID Phase 3 adds deterministic preprocessing from Phase 2-valid PCM16 WAV
input to an in-memory float32 mono 16000 Hz waveform.

The implementation uses:

- `numpy` for typed array operations and PCM16 to float32 conversion;
- `scipy.signal.resample_poly` for deterministic polyphase resampling;
- Phase 2 validation policy as the input gate;
- one open file snapshot for Phase 3 header validation, signal statistics, and
  PCM decode;
- a typed `PreprocessedAudioResult` that keeps waveform data out of public
  `repr()` and `to_dict()` output;
- a framework-neutral application service in the existing modular monolith.

The approved pipeline is:

```text
Phase 2 valid PCM16 WAV
-> decode PCM16
-> float32 via sample / 32768.0
-> stereo arithmetic mean downmix
-> DC offset removal
-> resample to 16000 Hz with scipy.signal.resample_poly
-> safety clipping and invariant validation
-> PreprocessedAudioResult
```

## Alternatives

1. Continue using only the Python standard library.
2. Add `librosa`, `soundfile`, `torchaudio`, PyTorch, SpeechBrain, or pyannote.
3. Use FFT-based resampling.
4. Delay downmix and resampling until the embedding backend is selected.
5. Add loudness, RMS, or peak normalization during preprocessing.

## Reason

Phase 3 needs array math and sample-rate conversion but still must avoid ML,
model downloads, API work, and broad audio frameworks. `numpy` is the minimal
foundation for deterministic numeric arrays, and `scipy.signal.resample_poly`
provides a stable polyphase resampling primitive with anti-aliasing behavior
appropriate for offline preprocessing.

Phase 3 must also avoid mixing validation metadata from one path state with
waveform data from another path state. Opening the file once and applying Phase
2 policy to the header and signal statistics read from that open snapshot keeps
source metadata and decoded waveform aligned without adding content
fingerprinting to the MVP scope.

`resample_poly` also lets the implementation pass reduced `up` and `down`
factors derived from `gcd(input_rate, 16000)`, which keeps the conversion
explicit and deterministic for the five Phase 2-supported sample rates.

Keeping preprocessing behind the existing application service boundary avoids
parallel architecture and preserves the modular monolith chosen in ADR-001.

## Consequences

- Phase 3 introduces the first runtime dependencies: `numpy>=1.26,<3` and
  `scipy>=1.11,<2`.
- Valid preprocessing results hold waveform data in memory, but public
  dictionaries, representations, errors, and logs do not expose waveform values
  or absolute paths.
- Stereo input is downmixed with arithmetic mean; mono input is not downmixed.
- Input audio files are read but not modified, copied, stored, or converted on
  disk.
- Safety clipping guards numeric invariants but is not normalization.
- A pathname replacement after the file is opened does not change the snapshot
  used by the current preprocessing operation on platforms with normal
  descriptor/rename semantics.
- `VALID` can still contain a zero waveform after arithmetic mean downmix or DC
  offset removal; post-preprocessing energy policy is explicitly deferred.
- VAD, silence trimming, denoising, embeddings, scoring, and match decisions
  remain deferred.

## Notes

Technical validation and deterministic preprocessing do not guarantee
suitability for speaker verification. Future embedding and scoring phases must
continue to avoid presenting similarity as probability unless calibrated on a
labeled dataset.

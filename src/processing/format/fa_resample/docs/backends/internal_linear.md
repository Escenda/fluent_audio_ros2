# internal_linear

`fa_resample` currently uses an internal linear interpolation resampler.
It has no external DSP backend.

## Input Contract

- input `AudioFrame`
- positive sample rate
- positive channel count
- `FLOAT32LE` encoding
- 32-bit sample depth
- `interleaved` layout
- finite normalized samples in `[-1.0, 1.0]`
- enabled mic/ref stream topics must be explicitly configured

## Output Contract

The node emits `FLOAT32LE` / 32-bit / interleaved frames at the configured
`target_sample_rate`. The current design requires `target_sample_rate=16000`.

## Failure Policy

Invalid frames are dropped. The node does not infer missing topics, silently
enable disabled streams, or choose another target sample rate. Unsupported
input format is a drop condition, not a hidden conversion fallback.

The backend does not decode PCM16/PCM32, does not encode PCM16, and does not
clamp overflow samples. Those operations belong to explicit format/dynamics
nodes in the system pipeline.

If a higher quality resampler is introduced later, it must be represented as an
explicit backend or package contract.

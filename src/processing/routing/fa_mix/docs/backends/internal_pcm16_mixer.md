# internal_pcm16_mixer

`fa_mix` currently uses an internal PCM16 mixer. It has no external routing or
DSP backend.

## Input Contract

- `input_topics` must contain at least one topic.
- `master_index` must refer to an input topic.
- frames must match configured sample rate, channels, bit depth, and encoding.
- current implementation requires `expected.bit_depth=16`.

## Output Contract

The node mixes fresh frames using explicit `input_gains_db` and publishes a
PCM16 `AudioFrame` on `output_topic`.

## Failure Policy

Missing inputs, stale frames, mismatched formats, unsupported bit depth, and
decode failures are not corrected implicitly. They are handled by drop/error
paths according to the node specification.

Future bus/router engines must be added as explicit backend contracts rather
than hidden runtime selection.

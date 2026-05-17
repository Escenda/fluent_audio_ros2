# internal_pcm16_mixer

`fa_mix` currently uses an internal PCM16 mixer. It has no external routing or
DSP backend.

## Input Contract

- `input_topics` must contain at least one topic.
- `input_gains_db` must contain one gain or one gain per input. Empty gain
  lists are invalid and are not treated as implicit 0 dB.
- `master_index` must refer to an input topic.
- frames must match configured sample rate, channels, bit depth, and encoding.
- frame `stream_id` must match the configured input topic.
- current implementation requires `expected.encoding=PCM16LE` and
  `expected.bit_depth=16`.

## Output Contract

The node mixes fresh frames using explicit `input_gains_db` and publishes a
PCM16 `AudioFrame` on `output_topic`.

## Failure Policy

Missing inputs, stale frames, mismatched frame lengths, mismatched formats,
unsupported PCM format, and decode failures are not corrected implicitly. The
node drops the entire mix instead of publishing a partial mix.

Future bus/router engines must be added as explicit backend contracts rather
than hidden runtime selection.

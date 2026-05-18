# internal_pcm16_mixer

`internal_pcm16_mixer` は `fa_mix_node` から分離された ROS-free C++ backend である。ROS2 topic、`AudioFrame`、diagnostics、publisher/subscriber、frame timestamp cache を知らない。

## Input Contract

- backend input buffer count must match configured gain count.
- every input buffer must contain PCM16LE interleaved bytes.
- `input_gains_db` must be explicit and finite. Empty gain lists are invalid and are not treated as implicit 0 dB.
- all decoded input buffers must have the same sample count.

## Output Contract

The backend returns a PCM16LE byte payload. The ROS node wraps that payload in an `AudioFrame`, sets `output.stream_id`, and publishes it on `output_topic`.

## Failure Policy

Mismatched buffer counts, empty inputs, misaligned inputs, mismatched sample counts, non-finite gains, and output overflow are not corrected implicitly. The backend returns a typed status and leaves the output payload unchanged.

Missing inputs, stale frames, topic/stream metadata mismatch, and timestamp validation are node responsibilities. Future bus/router engines must be added as explicit backend contracts rather than hidden runtime selection.

# phase_vocoder_pitch_shift backend

`phase_vocoder_pitch_shift` は phase vocoder based pitch transformation の backend contract である。

## Required Config

- `backend.name`
- `pitch.ratio` または `pitch.semitones`
- `frame_length`
- `hop_length`
- `window`
- `expected.sample_rate`
- `expected.channels`

## Forbidden

- hidden resample
- hidden trim / padding
- pitch estimate auto-detection
- voice conversion
- ROS2 topic/message dependency inside backend


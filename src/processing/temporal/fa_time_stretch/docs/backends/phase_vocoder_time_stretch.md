# phase_vocoder_time_stretch backend

`phase_vocoder_time_stretch` は phase vocoder based time-scale modification の backend contract である。

## Required Config

- `backend.name`
- `stretch.ratio`
- `frame_length`
- `hop_length`
- `window`
- `expected.sample_rate`
- `expected.channels`

## Forbidden

- hidden resample
- hidden trim / padding
- pitch shift
- timestamp compensation
- ROS2 topic/message dependency inside backend


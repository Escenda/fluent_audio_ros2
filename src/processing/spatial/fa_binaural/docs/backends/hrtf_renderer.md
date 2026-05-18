# hrtf_renderer backend

`hrtf_renderer` は HRTF profile に基づいて binaural rendering を行う backend contract である。

## Required Config

- `backend.name`
- `hrtf.profile_path`
- `input.layout`
- `input.channels`
- `output.channels`

## Forbidden

- missing HRTF profile fallback
- hidden stereo downmix
- hidden speaker output
- ROS2 topic/message dependency inside backend


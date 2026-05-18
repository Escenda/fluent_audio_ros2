# reverb_suppression backend

`reverb_suppression` は room profile または explicit model を使って dereverberation を行う backend contract である。

## Required Config

- `backend.name`
- `backend.execution`
- `room_profile_path` または `model_path`
- `expected.sample_rate`
- `expected.channels`

## Forbidden

- missing profile/model 時の pass-through
- hidden AEC
- hidden beamforming
- hidden source separation
- ROS2 topic/message dependency inside backend


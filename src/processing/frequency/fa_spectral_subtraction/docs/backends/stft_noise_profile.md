# stft_noise_profile backend

`stft_noise_profile` は STFT magnitude noise profile を使って spectral subtraction を行う backend contract である。

## Required Config

- `backend.name`
- `frame_length`
- `hop_length`
- `window`
- `noise_profile_path` または明示 calibration source
- `subtraction.amount`
- `subtraction.floor`

## Forbidden

- missing noise profile 時の pass-through
- hidden zero padding
- hidden truncation
- hidden gain compensation
- ROS2 topic/message dependency inside backend


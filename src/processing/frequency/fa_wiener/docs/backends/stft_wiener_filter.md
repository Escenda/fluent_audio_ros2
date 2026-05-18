# stft_wiener_filter backend

`stft_wiener_filter` は STFT power estimate に基づいて Wiener gain を適用する backend contract である。

## Required Config

- `backend.name`
- `frame_length`
- `hop_length`
- `window`
- `noise_estimate.source`
- `gain.floor`
- `gain.smoothing`

## Forbidden

- missing noise estimate 時の pass-through
- hidden zero padding
- hidden truncation
- hidden gain compensation
- ROS2 topic/message dependency inside backend


# wind_noise_reduction backend

`wind_noise_reduction` は wind-like burst / turbulence component を検出し、設定された suppression を適用する backend contract である。

## Required Config

- `backend.name`
- `detector.band_hz`
- `detector.threshold`
- `suppression.amount`
- `attack_ms`
- `release_ms`

## Forbidden

- hidden high-pass fallback
- hidden noise gate fallback
- hidden beamforming
- ROS2 topic/message dependency inside backend


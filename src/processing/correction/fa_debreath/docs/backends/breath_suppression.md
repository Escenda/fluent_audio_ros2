# breath_suppression backend

`breath_suppression` は breath-like noise を検出し、設定された attenuation を適用する backend contract である。

## Required Config

- `backend.name`
- `detection.band_hz`
- `detection.threshold`
- `attenuation.db`
- `attack_ms`
- `release_ms`

## Forbidden

- hidden VAD
- hidden normalize / compressor
- breath boundary guessing beyond configured detector
- ROS2 topic/message dependency inside backend


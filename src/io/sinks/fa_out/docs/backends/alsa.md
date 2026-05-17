# ALSA Sink Backend

## Backend Name

`alsa_playback`

## Contract

ALSA playback device を明示 id で開き、validated PCM frame を device へ書き込みます。

## Input

- ALSA playback device id
- expected sample rate
- expected channels
- expected bit depth
- PCM frame bytes

## Output

- device playback
- playback completion event

## Failure Conditions

- device open failure
- unsupported format
- unsupported sample rate
- unsupported channel count
- invalid queue / QoS / chunk config

失敗時に別 device へ暗黙 fallback しません。

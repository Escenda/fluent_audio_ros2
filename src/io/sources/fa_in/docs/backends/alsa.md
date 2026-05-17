# ALSA Source Backend

## Backend Name

`alsa_capture`

## Contract

`backend.name=alsa_capture` の backend です。ALSA capture source を明示 id または index で開き、PCM frame を `fa_in` に返します。

## Input

- ALSA PCM source id
- sample rate
- channels
- bit depth
- chunk duration

## Output

- interleaved PCM bytes
- format metadata

## Failure Conditions

- source enumeration failure
- configured source missing
- open failure
- unsupported format
- unsupported sample rate
- unsupported channel count

失敗時に `default` device へ暗黙 fallback しません。

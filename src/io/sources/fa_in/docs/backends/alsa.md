# ALSA Source Backend

## Backend Name

`alsa_capture`

## Contract

`backend.name=alsa_capture` の backend です。ALSA raw hardware capture source を明示 id または index で開き、PCM frame を `fa_in` に返します。

## Input

- ALSA PCM source id。`hw:` で始まる raw hardware source のみ許可する
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
- configured source が ALSA plugin PCM (`default`, `plug*`, `plughw*`, `sysdefault`, `pulse`, `pipewire` など)
- open failure
- unsupported format
- unsupported sample rate
- unsupported channel count

失敗時に `default` device へ暗黙 fallback しません。`default` や `plug*` は ALSA plugin 経由の暗黙変換を隠すため、この backend の候補一覧にも出しません。

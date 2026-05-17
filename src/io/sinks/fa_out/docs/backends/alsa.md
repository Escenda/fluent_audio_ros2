# ALSA Sink Backend

## Backend Name

`alsa_playback`

## Contract

`backend.name=alsa_playback` の backend です。ALSA raw hardware playback device を明示 id で開き、validated PCM frame を device へ書き込みます。

## Input

- ALSA playback device id。`hw:` で始まる raw hardware device のみ許可する
- expected sample rate
- expected channels
- expected bit depth。現行 backend は `PCM16LE/16` のみ対応
- PCM frame bytes

## Output

- device playback
- playback completion event

## Failure Conditions

- device open failure
- configured device が ALSA plugin PCM (`default`, `plug*`, `plughw*`, `sysdefault`, `pulse`, `pipewire` など)
- unsupported format
- bit depth が `16` ではない
- unsupported sample rate
- unsupported channel count
- runtime `snd_pcm_writei` XRUN / EAGAIN / error / zero-frame write
- runtime playback handle missing
- invalid queue / QoS / chunk config

失敗時に別 device へ暗黙 fallback しません。`default` や `plug*` は ALSA plugin 経由の暗黙変換を隠すため、この backend では指定できません。

runtime write failure は sink path の破断として扱います。device reopen や `snd_pcm_prepare` retry で継続せず、node を fail closed します。

# ALSA Source Backend

## Backend Name

`alsa_capture`

## Contract

`backend.name=alsa_capture` の backend です。ALSA raw hardware capture source を明示 id または index で開き、PCM frame を `fa_in` に返します。

## Input

- ALSA PCM source id。`hw:` で始まる raw hardware source のみ許可する
- sample rate
- channels
- encoding / bit depth pair。許可する組は `PCM16LE/16`, `PCM32LE/32`, `FLOAT32LE/32`
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
- unsupported encoding / bit depth pair
- unsupported sample rate
- unsupported channel count
- runtime `snd_pcm_readi` XRUN / error / zero-frame read

失敗時に `default` device へ暗黙 fallback しません。`default` や `plug*` は ALSA plugin 経由の暗黙変換を隠すため、この backend の候補一覧にも出しません。

runtime read failure は source path の破断として扱います。`snd_pcm_prepare` で継続したり、別 source を reopen したりせず、node を fail closed します。

ALSA software resampling は無効化します。指定 rate を device が受け付けない場合は fail closed し、backend 内で暗黙変換しません。

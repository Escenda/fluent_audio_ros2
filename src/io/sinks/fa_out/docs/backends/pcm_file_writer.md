# PCM File Writer Backend

## Backend Name

`pcm_file_writer`

## Contract

`backend.name=pcm_file_writer` の backend です。`fa_out` が受け付けた
`AudioFrame` payload を、明示された file target へ raw PCM bytes として
write します。

backend は ROS2 adapter ではないため、ROS2 API や ROS message 型に依存しません。
ROS topic、stream identity、playback lifecycle は `fa_out` node が扱います。

## Supported AudioFrame / Backend Capability

- `encoding` / `bit_depth`: `PCM16LE/16`、`PCM32LE/32`、`FLOAT32LE/32`。
- `sample_rate`: `fa_out` node が configured `audio.sample_rate` と frame metadata の一致を検証する。backend は file に sample rate を推定・変換しない。
- `channels`: positive configured channel count。byte count は `channels * bit_depth / 8` で計算する。
- `layout`: `interleaved`。non-interleaved frame は `fa_out` node が reject し、backend は downmix/upmix/deinterleave しない。
- file contract: target file は accepted `AudioFrame.data` の raw PCM bytes を順に保持する。codec header、container encode、parent directory creation は行わない。

unsupported frame / config は frame reject、startup fail、runtime fatal、または explicit backend error result にする。hidden encode、resample、downmix、channel conversion、bit-depth conversion、gain、normalize、limiter は行わない。

## Required Config

- `file.path`
- `overwrite.enabled`
- `audio.encoding`
- `audio.channels`
- `audio.bit_depth`

`PCM16LE/16`、`PCM32LE/32`、`FLOAT32LE/32` だけを受け付けます。

## Forbidden

- hidden codec encode
- hidden resample
- hidden channel conversion
- hidden gain / normalize / limiter
- output path guessing
- parent directory creation

## Failure Conditions

- empty `file.path`
- missing parent directory
- target exists while `overwrite.enabled=false`
- directory target
- unsupported encoding / bit depth pair
- write error

# PCM File Reader Backend

## Backend Name

`pcm_file_reader`

## Contract

`backend.name=pcm_file_reader` は raw PCM file を `fa_in` の source backend として読む。encoded file decode、resample、channel conversion、bit-depth conversion、gain、normalize は行わない。

この backend は ROS-free である。filesystem / binary file read / EOF / loop だけを扱い、ROS2 topic、ROS message、`rclcpp` を知らない。

## Supported AudioFrame Capability

- `encoding`: headerless raw PCM として configured `audio.encoding` を metadata に使う。現行 executable test は `PCM16LE` を代表 format として検証する。
- `bit_depth`: positive かつ byte-aligned。`channels * bit_depth / 8` が file frame byte size になる。
- `sample_rate`: configured `audio.sample_rate` を publish metadata として使い、file から推定しない。
- `channels`: positive configured channel count。payload は configured layout の interleaved frame 列として扱う。
- `layout`: `interleaved`。non-interleaved file を暗黙に interleave/deinterleave しない。
- file contract: file は headerless raw PCM payload だけを含む。file size は expected frame byte size で割り切れる必要がある。

unsupported encoding / bit depth / sample_rate / channels / layout / file shape は startup fail または explicit read error にする。hidden decode、resample、downmix、channel conversion、bit-depth conversion は行わない。

## Required Parameters

- `file.path`: 既存 regular file。empty、missing、directory、empty file は fail closed。
- `audio.source_id`: publish する `AudioFrame.source_id`。empty は fail closed。
- `audio.sample_rate`
- `audio.channels`
- `audio.bit_depth`
- `audio.encoding`
- `audio.layout`
- `audio.chunk_ms`
- `playback.loop`

`AudioFrame.stream_id` は `fa_in` node の `audio.stream_id` を使う。`file.path` を stream identity として流用しない。

## Failure Conditions

- unknown backend name
- empty / missing `file.path`
- non-regular file
- empty file
- file size が expected frame byte size で割り切れない
- null destination
- requested frames / bytes が 0
- read error
- partial byte frame

EOF は `playback.loop=false` の場合に source completion として扱う。別 source への fallback はしない。`playback.loop=true` の場合は明示設定に基づき file 先頭へ戻る。

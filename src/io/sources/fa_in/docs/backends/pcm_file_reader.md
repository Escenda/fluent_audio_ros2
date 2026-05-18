# PCM File Reader Backend

## Backend Name

`pcm_file_reader`

## Contract

`backend.name=pcm_file_reader` は raw PCM file を `fa_in` の source backend として読む。encoded file decode、resample、channel conversion、bit-depth conversion、gain、normalize は行わない。

この backend は ROS-free である。filesystem / binary file read / EOF / loop だけを扱い、ROS2 topic、ROS message、`rclcpp` を知らない。

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

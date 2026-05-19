# network_pcm_sender backend

`network_pcm_sender` は `fa_out` の raw PCM UDP sink backend です。

必須 config:

- `endpoint.uri`: 送信先 `udp://IPv4:port`
- `transport.identity`: transport identity
- `network.max_packet_bytes`: 1 UDP packet の最大 byte 数

backend は POSIX socket と raw byte buffer だけを扱い、ROS2 topic、ROS2 message、`rclcpp` を知りません。`fa_out` は accepted `AudioFrame` 1 件を 1 UDP packet として backend に渡します。encode、resample、gain、jitter buffer、PLC、clock drift correction は行いません。

Supported AudioFrame / packet capability:

- `encoding` / `bit_depth`: `PCM16LE/16`、`PCM32LE/32`、`FLOAT32LE/32`。
- `sample_rate`: `fa_out` node が configured `audio.sample_rate` と frame metadata の一致を検証する。backend は packet payload の sample rate を推定・変換しない。
- `channels`: positive configured channel count。byte count は `channels * bit_depth / 8` で計算する。
- `layout`: `interleaved`。non-interleaved frame は `fa_out` node が reject し、backend は downmix/upmix/deinterleave しない。
- packet contract: accepted `AudioFrame` 1 件は UDP packet 1 件になる。packet byte count は `network.max_packet_bytes` 以下で、expected frame byte size で割り切れる必要がある。

unsupported endpoint / frame / config は startup fail、frame reject、runtime fatal、または explicit backend error result にする。hidden encode、resample、downmix、format conversion、jitter buffer、PLC、clock drift correction は行わない。

Fail closed 条件:

- `endpoint.uri` が空、`udp://IPv4:port` でない、host が IPv4 address でない、port が空 / 非数 / `0` / `65535` 超過
- `network.max_packet_bytes <= 0`
- `network.max_packet_bytes` が expected frame byte size で割り切れない
- `writeFrames()` に渡された frame count が `0`
- `writeFrames()` の byte count が `network.max_packet_bytes` を超える
- UDP socket open / send が失敗する

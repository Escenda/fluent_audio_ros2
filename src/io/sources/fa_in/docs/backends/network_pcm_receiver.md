# network_pcm_receiver backend

`network_pcm_receiver` は `fa_in` の raw PCM UDP source backend です。

必須 config:

- `endpoint.uri`: bind する `udp://IPv4:port`
- `transport.identity`: transport identity
- `network.max_packet_bytes`: 1 UDP packet の最大 byte 数
- `polling.period_ms`: packet 未到着時の poll interval
- `network.source_timeout_ms`: packet 未到着を許容する最大時間

backend は POSIX socket と raw byte buffer だけを扱い、ROS2 topic、ROS2 message、`rclcpp` を知りません。受信 payload は expected frame byte size で割り切れる場合だけ `AudioFrame.data` へ渡されます。decode、resample、gain、jitter buffer、PLC、clock drift correction は行いません。

Supported AudioFrame / packet capability:

- `encoding`: raw PCM として configured `audio.encoding` を metadata に使う。現行 executable test は `PCM16LE` を代表 format として検証する。
- `bit_depth`: positive かつ byte-aligned。`channels * bit_depth / 8` が packet frame byte size になる。
- `sample_rate`: configured `audio.sample_rate` を publish metadata として使い、network payload から推定しない。
- `channels`: positive configured channel count。payload は configured layout の interleaved frame 列として扱う。
- `layout`: `interleaved`。non-interleaved packet を暗黙に interleave/deinterleave しない。
- packet contract: accepted UDP packet は 1 `AudioFrame.data` になる。packet は empty ではなく、`network.max_packet_bytes` 以下で、expected frame byte size で割り切れる必要がある。

unsupported endpoint / encoding / bit depth / sample_rate / channels / layout / packet shape は startup fail、runtime fatal、または explicit read error にする。hidden resample、downmix、format conversion、jitter buffer、PLC、clock drift correction は行わない。

Fail closed / explicit status 条件:

- `endpoint.uri` が空、`udp://IPv4:port` でない、host が IPv4 address でない、port が空 / 非数 / `0` / `65535` 超過
- open 時の requested frame count が `0`
- open 時の audio format が byte aligned でない
- UDP socket open / bind が失敗する
- `read()` に渡された destination が null、または frame count が `0`
- 受信 packet が空、`network.max_packet_bytes` を超える、または expected frame byte size で割り切れない
- node 側で `network.source_timeout_ms` を超えて packet が未着になる

packet 未到着は backend 単体では `ReadStatus::kNoData` として返します。`fa_in` node は `network_pcm_receiver` の場合だけ同一 endpoint を `polling.period_ms` 待って再 poll しますが、`network.source_timeout_ms` を超えた未着は required source の欠落として fail closed します。別 source への fallback や retry hidden recovery は行いません。

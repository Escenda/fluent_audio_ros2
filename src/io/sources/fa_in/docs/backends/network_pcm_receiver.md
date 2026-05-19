# network_pcm_receiver backend

`network_pcm_receiver` は `fa_in` の raw PCM UDP source backend です。

必須 config:

- `endpoint.uri`: bind する `udp://IPv4:port`
- `transport.identity`: transport identity
- `network.max_packet_bytes`: 1 UDP packet の最大 byte 数
- `polling.period_ms`: packet 未到着時の poll interval

backend は POSIX socket と raw byte buffer だけを扱い、ROS2 topic、ROS2 message、`rclcpp` を知りません。受信 payload は expected frame byte size で割り切れる場合だけ `AudioFrame.data` へ渡されます。decode、resample、gain、jitter buffer、PLC、clock drift correction は行いません。

Fail closed / explicit status 条件:

- `endpoint.uri` が空、`udp://IPv4:port` でない、host が IPv4 address でない、port が空 / 非数 / `0` / `65535` 超過
- open 時の requested frame count が `0`
- open 時の audio format が byte aligned でない
- UDP socket open / bind が失敗する
- `read()` に渡された destination が null、または frame count が `0`
- 受信 packet が空、`network.max_packet_bytes` を超える、または expected frame byte size で割り切れない

packet 未到着は failure ではなく `ReadStatus::kNoData` として返します。`fa_in` node は `network_pcm_receiver` の場合だけ同一 endpoint を `polling.period_ms` 待って再 poll します。別 source への fallback や retry hidden recovery は行いません。

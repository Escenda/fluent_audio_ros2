# network_pcm_receiver backend

`network_pcm_receiver` は `fa_in` の raw PCM UDP source backend です。

必須 config:

- `endpoint.uri`: bind する `udp://IPv4:port`
- `transport.identity`: transport identity
- `network.max_packet_bytes`: 1 UDP packet の最大 byte 数
- `polling.period_ms`: packet 未到着時の poll interval

backend は POSIX socket と raw byte buffer だけを扱い、ROS2 topic、ROS2 message、`rclcpp` を知りません。受信 payload は expected frame byte size で割り切れる場合だけ `AudioFrame.data` へ渡されます。decode、resample、gain、jitter buffer、PLC、clock drift correction は行いません。

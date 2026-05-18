# network_pcm_sender backend

`network_pcm_sender` は `fa_out` の raw PCM UDP sink backend です。

必須 config:

- `endpoint.uri`: 送信先 `udp://IPv4:port`
- `transport.identity`: transport identity
- `network.max_packet_bytes`: 1 UDP packet の最大 byte 数

backend は POSIX socket と raw byte buffer だけを扱い、ROS2 topic、ROS2 message、`rclcpp` を知りません。`fa_out` は accepted `AudioFrame` 1 件を 1 UDP packet として backend に渡します。encode、resample、gain、jitter buffer、PLC、clock drift correction は行いません。

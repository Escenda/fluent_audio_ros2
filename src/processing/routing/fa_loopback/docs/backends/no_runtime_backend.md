# no runtime backend

`fa_loopback` は外部 runtime backend を持たない。

処理は ROS2 subscription callback 内で完結する。入力 `fa_interfaces/msg/AudioFrame` を copy し、`stream_id` だけを `output_topic` に更新して publish する。

依存する runtime data:

- `fa_interfaces/msg/AudioFrame`
- ROS2 topic graph
- ROS2 parameters

外部 DSP library、device API、model runtime、cloud API には接続しない。

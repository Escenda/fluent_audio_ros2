# no runtime backend

`fa_loopback` は runtime backend を持たない。

この package は byte-for-byte routing node であり、音声処理 engine を選択しない。入力 `fa_interfaces/msg/AudioFrame` を copy し、`stream_id` だけを `output.stream_id` に更新して publish する。

依存する runtime data:

- `fa_interfaces/msg/AudioFrame`
- ROS2 topic graph
- ROS2 parameters

外部 DSP library、device API、model runtime、cloud API には接続しない。

topic 名と stream ID は別契約である。ROS2 topic graph は配送経路を決め、`AudioFrame.stream_id` は stream identity を表す。

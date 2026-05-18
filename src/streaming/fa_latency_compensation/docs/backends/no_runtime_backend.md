# No Runtime Backend

`fa_latency_compensation` は外部 runtime backend を持ちません。

ROS2 callback 内で `AudioFrame` metadata を検証し、`header.stamp` と `stream_id` のみを更新します。入力 `stream_id` は `input_stream_id`、出力 `stream_id` は `output.stream_id` であり、ROS topic 名から推測しません。device I/O、DSP backend、model backend、network backend はこの package の責務外です。

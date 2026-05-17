# No Runtime Backend

`fa_latency_compensation` は外部 runtime backend を持ちません。

ROS2 callback 内で `AudioFrame` metadata を検証し、`header.stamp` と `stream_id` のみを更新します。device I/O、DSP backend、model backend、network backend はこの package の責務外です。

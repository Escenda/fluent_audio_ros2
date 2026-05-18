# fa_aec_linear

線形 AEC（Acoustic Echo Cancellation）ノードです。現状は **参照を単純減算**する baseline で、論文ベース実装へ差し替える前提です。サンプル処理は ROS2 を知らない `baseline_linear` backend に閉じ、node は topic/cache/diagnostics を担当します。

## Subscribe / Publish
- Sub:
  - `audio/resample16k/mic`（`fa_interfaces/msg/AudioFrame`）
  - `audio/resample16k/ref`（`fa_interfaces/msg/AudioFrame`）
- Pub:
  - `audio/aec_linear/frame`（`fa_interfaces/msg/AudioFrame`）

## Run
```bash
ros2 launch fa_aec_linear fa_aec_linear.launch.py
```

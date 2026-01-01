# fa_resample

FluentAudio の 16k ストリームを集約して供給するリサンプルノードです（設計上 `target_sample_rate=16000` 固定）。

## Publish
- `audio/resample16k/mic`（`fa_interfaces/msg/AudioFrame`）
- `audio/resample16k/ref`（`fa_interfaces/msg/AudioFrame`）※任意

## Subscribe
- `audio/frame`（`fa_interfaces/msg/AudioFrame`）
- `audio/output/frame`（`fa_interfaces/msg/AudioFrame`）※任意（AECの参照用）

## Run
```bash
ros2 launch fa_resample fa_resample.launch.py
```

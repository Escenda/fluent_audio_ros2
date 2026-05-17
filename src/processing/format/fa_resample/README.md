# fa_resample

FluentAudio の `FLOAT32LE` / 32-bit / interleaved audio frame を 16kHz へ変換するリサンプルノードです（設計上 `target_sample_rate=16000` 固定）。

PCM16 / PCM32 から float32 への変換は `fa_sample_format` に明示的に任せます。`fa_resample` は sample format conversion、bit depth conversion、clamp、gain、channel conversion を行いません。

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

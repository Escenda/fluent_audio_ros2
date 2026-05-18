# fa_resample

FluentAudio の `FLOAT32LE` / 32-bit / interleaved audio frame を `target_sample_rate` へ変換するリサンプルノードです。

PCM16 / PCM32 から float32 への変換は `fa_sample_format` に明示的に任せます。`fa_resample` は sample format conversion、bit depth conversion、clamp、gain、channel conversion を行いません。

## Publish
- `audio/resample16k/mic`（`fa_interfaces/msg/AudioFrame`）
- `audio/resample16k/ref`（`fa_interfaces/msg/AudioFrame`）※任意
- output stream identity は `mic.output.stream_id` / `ref.output.stream_id`

## Subscribe
- `audio/frame`（`fa_interfaces/msg/AudioFrame`）
- `audio/output/frame`（`fa_interfaces/msg/AudioFrame`）※任意（AECの参照用）
- input stream identity は `mic.input_stream_id` / `ref.input_stream_id`

ROS topic 名と `AudioFrame.stream_id` は別契約です。topic 名を stream identity として代用する設定は起動失敗します。

## Run
```bash
ros2 launch fa_resample fa_resample.launch.py
```

# fa_mix

複数入力のミキシング（MVP）ノードです。現状は **PCM16LE の加算ミックス + ゲイン**のみを実装しています。
configured input が欠ける、stale、decode 不能、または sample count mismatch の場合は partial mix を publish せず frame 全体を drop します。

## Subscribe / Publish
- Sub: `input_topics`（`fa_interfaces/msg/AudioFrame`）
- Pub: `audio/output/frame`（`fa_interfaces/msg/AudioFrame`）※既定

## Run
```bash
ros2 launch fa_mix fa_mix.launch.py
```

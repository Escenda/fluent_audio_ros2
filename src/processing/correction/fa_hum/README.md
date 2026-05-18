# fa_hum

`fa_hum` は `fa_interfaces/msg/AudioFrame` の `FLOAT32LE` interleaved stream から、電源 hum とその倍音を notch cascade で除去する ROS 2 correction package です。

## 入出力契約

- 入力: `FLOAT32LE`, `bit_depth=32`, `layout=interleaved`
- サンプル範囲: normalized `[-1.0, 1.0]`
- 出力: metadata は入力を保持し、`stream_id` と `data` のみを更新
- 入力 `stream_id`: `input_stream_id` と一致必須
- 出力 `stream_id`: `output.stream_id` に更新
- ROS topic (`input_topic` / `output_topic`) と AudioFrame stream identity は別の識別子として扱う
- invalid frame / invalid sample: publish せず drop
- source change: filter state をリセットして新しい `source_id` で処理を継続

## 設定

`config/default.yaml`:

```yaml
fa_hum:
  ros__parameters:
    input_topic: "fa_hum/input"
    output_topic: "fa_hum/output"
    input_stream_id: "audio/dc_offset_removed/mic"
    output:
      stream_id: "audio/hum_removed/mic"
    hum:
      frequency_hz: 60.0
      harmonics: 4
      q: 30.0
```

`hum.frequency_hz` は有限かつ `> 0` の値を受け付けます。`hum.harmonics` は `>= 1`、`hum.q` は有限かつ `> 0` が必須です。実際に構成される notch stage は `frequency_hz * n < sample_rate / 2` を満たす倍音だけです。

`input_topic` / `output_topic` は ROS2 の配送先 topic です。`input_stream_id` / `output.stream_id` は `AudioFrame.stream_id` の信号系列 ID であり、raw/resolved ROS topic と一致してはなりません。`input_stream_id` と `output.stream_id` も同一にできません。

## 起動

```bash
ros2 launch fa_hum fa_hum.launch.py
```

## 検証

```bash
python3 -m pytest src/processing/correction/fa_hum/test/unit
```

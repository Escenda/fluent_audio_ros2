# fa_silence_removal

`fa_silence_removal` は FluentAudio の FLOAT32LE interleaved `AudioFrame` chunk を RMS で判定し、silent chunk を publish せずに drop する ROS2 C++ パッケージです。

## 契約

- 入力: `input_topic` の `fa_interfaces/msg/AudioFrame`
- 出力: `output_topic` の `fa_interfaces/msg/AudioFrame`
- 対応形式: `sample_rate > 0`, `channels > 0`, `encoding=FLOAT32LE`, `bit_depth=32`, `layout=interleaved`
- `source_id` と `stream_id` は必須
- 入力 `stream_id` は `input_topic` と一致必須
- 出力 `stream_id` は `output_topic` に更新
- 入力サンプルは有限な正規化 FLOAT32 `[-1.0, 1.0]`
- `threshold.rms` 以上の chunk は non-silent として publish
- `threshold.rms` 未満の chunk は silent として drop
- speech activity 後は `hangover_ms` 分だけ silent chunk も publish する

## パラメータ

| 名前 | 既定値 | 内容 |
| --- | --- | --- |
| `input_topic` | `audio/buffered/mic` | 入力 AudioFrame topic |
| `output_topic` | `audio/silence_removed/mic` | 出力 AudioFrame topic |
| `threshold.rms` | `0.02` | silent 判定 RMS 閾値 |
| `hangover_ms` | `200.0` | activity 後に silent chunk を publish し続ける時間 |
| `expected.sample_rate` | `16000` | 入力 sample rate |
| `expected.channels` | `1` | 入力 channel 数 |
| `expected.encoding` | `FLOAT32LE` | 入力 encoding |
| `expected.bit_depth` | `32` | 入力 bit depth |
| `expected.layout` | `interleaved` | 入力 layout |
| `qos.depth` | `10` | AudioFrame QoS depth |
| `qos.reliable` | `false` | `true` で reliable、`false` で best effort |
| `diagnostics.publish_period_ms` | `1000` | diagnostics 発行周期 |

## 非責務

このパッケージは silent chunk removal のみを行います。silent sample の zeroing、gain gate、noise gate、normalize、resampling、format conversion、channel conversion、device I/O は行いません。

## 検証

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider src/processing/temporal/fa_silence_removal/test/unit -q
```

# fa_declick

`fa_declick` は FluentAudio の FLOAT32LE interleaved `AudioFrame` に含まれる短い impulse click を、前後サンプルから明示補正する ROS2 C++ パッケージです。

## 契約

- 入力: `input_topic` の `fa_interfaces/msg/AudioFrame`
- 出力: `output_topic` の `fa_interfaces/msg/AudioFrame`
- 対応形式: `sample_rate > 0`, `channels > 0`, `encoding=FLOAT32LE`, `bit_depth=32`, `layout=interleaved`
- `threshold.delta` は click 判定の差分しきい値
- `window.max_samples` は 1 回の補正で扱う連続 click sample 数の上限
- `source_id` と `stream_id` は必須
- 入力 `stream_id` は `input_stream_id` と一致必須
- 出力 `stream_id` は `output.stream_id` に更新
- 入力サンプルは有限な正規化 FLOAT32 `[-1.0, 1.0]`
- 出力サンプルが正規化範囲外になる場合は clamp せず frame を drop
- サンプル処理は ROS2 を知らない `internal_impulse_declick` backend に閉じる
- resolved `input_topic` と `output_topic` が一致する設定は起動失敗
- `input_stream_id` / `output.stream_id` は ROS topic identity および相互に distinct 必須

## パラメータ

| 名前 | 既定値 | 内容 |
| --- | --- | --- |
| `input_topic` | `fa_declick/input` | 入力 AudioFrame topic |
| `output_topic` | `fa_declick/output` | 出力 AudioFrame topic |
| `input_stream_id` | `audio/noise_gated/mic` | 入力 AudioFrame stream identity |
| `output.stream_id` | `audio/declicked/mic` | 出力 AudioFrame stream identity |
| `threshold.delta` | `0.25` | click 判定に使うサンプル差分 |
| `window.max_samples` | `1` | 連続 impulse run の最大 sample 数 |
| `expected.sample_rate` | `16000` | 入力 sample rate |
| `expected.channels` | `1` | 入力 channel 数 |
| `expected.encoding` | `FLOAT32LE` | 入力 encoding |
| `expected.bit_depth` | `32` | 入力 bit depth |
| `expected.layout` | `interleaved` | 入力 layout |
| `qos.depth` | `10` | AudioFrame QoS depth |
| `qos.reliable` | `false` | `true` で reliable、`false` で best effort |
| `diagnostics.publish_period_ms` | `1000` | diagnostics 発行周期 |

## 非責務

このパッケージは impulse declick correction のみを行います。device I/O、resampling、sample format conversion、channel conversion、gain、normalize、limiter、filter、denoise、declip、decrackle、echo、reverb は行いません。

## 検証

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider src/processing/correction/fa_declick/test/unit src/processing/correction/fa_declick/test/launch -q
```

# fa_delay

`fa_delay` は FluentAudio の FLOAT32LE interleaved `AudioFrame` に、設定した時間分の intentional temporal delay を適用する ROS2 C++ パッケージです。

## 契約

- 入力: `input_topic` の `fa_interfaces/msg/AudioFrame`
- 出力: `output_topic` の `fa_interfaces/msg/AudioFrame`
- 対応形式: `sample_rate > 0`, `channels > 0`, `encoding=FLOAT32LE`, `bit_depth=32`, `layout=interleaved`
- `delay.ms` は `expected.sample_rate` から whole samples に変換する
- `source_id` と `stream_id` は必須
- 入力 `stream_id` は `input_stream_id` と一致必須
- 出力 `stream_id` は `output.stream_id` に更新
- 入力サンプルは有限な正規化 FLOAT32 `[-1.0, 1.0]`
- `source_id` が変わった accepted frame では delay buffer をリセットし、設定 delay 分の silence を再度先頭に挿入する

## パラメータ

`fa_delay.launch.py` は `config_file` を必須 launch argument として扱い、package 内の設定を暗黙には読み込まない。以下は `config/default.yaml` の設定例であり、runtime default ではない。

| 名前 | 設定例 | 内容 |
| --- | --- | --- |
| `input_topic` | `fa_delay/input` | 入力 AudioFrame topic |
| `output_topic` | `fa_delay/output` | 出力 AudioFrame topic |
| `input_stream_id` | `audio/buffered/mic` | 入力 stream identity |
| `output.stream_id` | `audio/delayed/mic` | 出力 stream identity |
| `delay.ms` | `250.0` | 遅延時間。`expected.sample_rate` で whole samples へ丸める |
| `expected.sample_rate` | `16000` | 入力 sample rate |
| `expected.channels` | `1` | 入力 channel 数 |
| `expected.encoding` | `FLOAT32LE` | 入力 encoding |
| `expected.bit_depth` | `32` | 入力 bit depth |
| `expected.layout` | `interleaved` | 入力 layout |
| `qos.depth` | `10` | AudioFrame QoS depth |
| `qos.reliable` | `false` | `true` で reliable、`false` で best effort |
| `diagnostics.publish_period_ms` | `1000` | diagnostics 発行周期 |
| `diagnostics.qos.depth` | `10` | diagnostics QoS depth |
| `diagnostics.qos.reliable` | `true` | diagnostics を reliable にするか |

## 非責務

このパッケージは sample delay のみを行います。device I/O、resampling、sample format conversion、channel conversion、gain、normalize、limiter、filter、denoise、echo、reverb は行いません。

## 検証

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider src/processing/temporal/fa_delay/test/unit -q
```

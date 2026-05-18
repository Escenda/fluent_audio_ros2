# fa_fade

`fa_fade` は FluentAudio の FLOAT32LE interleaved `AudioFrame` に線形 fade-in / fade-out を適用する ROS2 C++ パッケージです。

## 契約

- 入力: `input_topic` の `fa_interfaces/msg/AudioFrame`
- 出力: `output_topic` の `fa_interfaces/msg/AudioFrame`
- 対応形式: `sample_rate > 0`, `channels > 0`, `encoding=FLOAT32LE`, `bit_depth=32`, `layout=interleaved`
- `source_id` と `stream_id` は必須
- 入力 `stream_id` は `input_stream_id` と一致必須
- 出力 `stream_id` は `output.stream_id` に更新
- 入力サンプルは有限な正規化 FLOAT32 `[-1.0, 1.0]`
- fade 計算と position state は ROS2 非依存の `internal_linear_fade` backend が担当する

## パラメータ

`fa_fade.launch.py` は `config_file` を必須 launch argument として扱い、package 内の設定を暗黙には読み込まない。以下は `config/default.yaml` の設定例であり、runtime default ではない。

| 名前 | 設定例 | 内容 |
| --- | --- | --- |
| `input_topic` | `fa_fade/input` | 入力 AudioFrame topic |
| `output_topic` | `fa_fade/output` | 出力 AudioFrame topic |
| `input_stream_id` | `audio/buffered/mic` | 入力 stream identity |
| `output.stream_id` | `audio/faded/mic` | 出力 stream identity |
| `fade.mode` | `fade_in` | `fade_in` または `fade_out` |
| `fade.duration_frames` | `16000` | fade 長。`> 0` |
| `fade.initial_position_frames` | `0` | 初期フレーム位置。`>= 0` |
| `qos.depth` | `10` | AudioFrame QoS depth |
| `qos.reliable` | `false` | `true` で reliable、`false` で best effort |
| `diagnostics.publish_period_ms` | `1000` | diagnostics 発行周期 |
| `diagnostics.qos.depth` | `10` | diagnostics QoS depth |
| `diagnostics.qos.reliable` | `true` | diagnostics を reliable にするか |

## 非責務

このパッケージは fade のみを行います。device I/O、resampling、sample format conversion、channel conversion、gain、normalize、limiter、filter、denoise は行いません。

## 検証

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider src/processing/temporal/fa_fade/test/unit -q
```

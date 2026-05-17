# fa_normalize

`fa_normalize` は `FLOAT32LE` interleaved `AudioFrame` stream に per-frame peak normalization を適用する dynamics processing node です。

- Sub: `audio/noise_gated/mic` (`fa_interfaces/msg/AudioFrame`)
- Pub: `audio/normalized/mic` (`fa_interfaces/msg/AudioFrame`)
- Executable: `fa_normalize_node`

この package は peak normalize のみを行います。device I/O、resampling、sample format conversion、channel conversion、compressor、limiter、gate、filter、denoise、LUFS/loudness normalize は行いません。

## Contract

入力 `AudioFrame` は次の契約を満たす必要があります。

- `source_id` は non-empty
- `stream_id` は non-empty かつ `input_topic` と一致
- `sample_rate > 0` かつ設定値と一致
- `channels > 0` かつ設定値と一致
- `encoding == FLOAT32LE`
- `bit_depth == 32`
- `layout == interleaved`
- `data` は non-empty で `channels * sizeof(float)` の倍数
- 全 sample は finite かつ normalized `[-1.0, 1.0]`

起動時設定が不正な場合は fail closed します。runtime frame が不正な場合は warning を出して publish せず drop します。

## Parameters

| name | default | rule |
| --- | ---: | --- |
| `input_topic` | `audio/noise_gated/mic` | non-empty |
| `output_topic` | `audio/normalized/mic` | non-empty |
| `normalize.target_peak_linear` | `0.9` | finite, `> 0.0`, `<= 1.0` |
| `normalize.silence_threshold_linear` | `0.0001` | finite, `>= 0.0`, `< target_peak_linear` |
| `expected.sample_rate` | `16000` | `> 0` |
| `expected.channels` | `1` | `> 0` |
| `expected.encoding` | `FLOAT32LE` | fixed |
| `expected.bit_depth` | `32` | fixed |
| `expected.layout` | `interleaved` | fixed |
| `qos.depth` | `10` | `> 0` |
| `qos.reliable` | `false` | best effort by default |
| `diagnostics.publish_period_ms` | `1000` | `> 0` |

## Diagnostics

`/diagnostics` に `target_peak_linear`、`silence_threshold_linear`、`last_gain`、`frames_in`、`frames_out`、`frames_dropped`、`frames_silence_passthrough`、`frames_normalized`、`output_topic` を publish します。

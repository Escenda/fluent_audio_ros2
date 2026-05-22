# fa_resample Backend Quality Requirements

この文書は Product Owner から Node Engineer と ClaudeCode Documentation Writer へ渡す要求定義である。
`fa_resample` の実装完了を示す資料ではない。

対象は `src/processing/format/fa_resample` である。

## 1. Product Goal

`fa_resample` は、音声内容をなるべく変えずに `AudioFrame.sample_rate` だけを変える format processing node として成立させる。

現状の `internal_linear_resampler` は軽量な内部実装として残してよいが、production voice pipeline の本命 backend として扱わない。

追加する backend は次の 2 系統とする。

| backend | 目的 | 位置づけ |
| --- | --- | --- |
| `speexdsp` | リアルタイム音声対話向けの低遅延 resampler | production realtime default 候補 |
| `soxr` | 高品質変換、録音、評価、非 realtime critical 処理向け resampler | high quality backend |

`internal_linear_resampler` は debug / reference / minimal backend として明示選択できるようにする。
missing backend を `internal_linear_resampler` に自動 fallback してはならない。

## 2. Responsibility Boundary

`fa_resample` は sample rate conversion だけを担当する。

以下は `fa_resample` の責務ではない。

- sample format conversion
- bit depth conversion
- channel conversion
- interleave / planar conversion
- gain / normalize / limiter / clamp
- denoise / AGC / model input

たとえば `PCM16LE` を `FLOAT32LE` にする処理は `fa_sample_format` の責務である。
`fa_resample` backend 内で暗黙に変換してはならない。

## 3. Backend Selection Contract

`backend.name` は明示必須とする。

想定値:

- `internal_linear_resampler`
- `speexdsp`
- `soxr`

unknown backend は startup failure とする。

`speexdsp` / `soxr` の runtime dependency が存在しない場合は、選択された backend の startup failure とする。
別 backend へ自動で切り替えてはならない。

## 4. Input / Output Capability

当面の `fa_resample` capability は以下を維持する。

```text
input:
  encoding: FLOAT32LE
  bit_depth: 32
  layout: interleaved
  channels: >= 1
  sample_rate: > 0

output:
  encoding: FLOAT32LE
  bit_depth: 32
  layout: interleaved
  channels: input channels と同一
  sample_rate: target_sample_rate
```

入力 `encoding`、`bit_depth`、`layout`、`channels`、payload alignment、sample 値域が capability 外の場合、backend state を触る前に frame rejection とする。

sample は finite かつ normalized range `[-1.0, 1.0]` を満たす必要がある。
clip / clamp / normalize で成功扱いにしてはならない。

## 5. Stream State Contract

resampler backend は stream ごとに state を持つ。

stream state の key は少なくとも以下で決まる。

- input stream identity
- input sample rate
- target sample rate
- channel count
- backend name
- backend quality parameter

同一 stream の途中で sample rate、channel count、encoding、layout が変わった場合は contract violation として扱う。
通常の microphone stream では起こらない前提だが、発生した場合に推測で継続してはならない。

streaming resampling では、chunk ごとに backend を再初期化してはならない。
fractional phase、filter history、pending output を backend state として保持する。

## 6. speexdsp Backend Requirement

`speexdsp` backend は real-time voice pipeline の標準候補とする。

設定項目の候補:

```yaml
backend.name: speexdsp
backend.quality: 6
```

`backend.quality` は SpeexDSP の quality range `0..10` に対応させる。

要求:

- quality は startup validation で `0..10` の integer に限定する。
- `speex_resampler_process_interleaved_float` 相当の interleaved float path を使う。
- mono / multi-channel の channel count を変えない。
- backend が保持する latency は SpeexDSP API から取得できる場合、diagnostics / test evidence に反映する。
- realtime voice 用 default 候補は quality `5` または `6` とし、最終値は実測で決める。

期待する性質:

- `internal_linear_resampler` より downsample aliasing が少ない。
- `soxr` より algorithmic delay / CPU cost を抑えやすい。
- KWS / Turn Detector / other model-input 前段で使いやすい。

## 7. soxr Backend Requirement

`soxr` backend は high-quality resampling backend とする。

設定項目の候補:

```yaml
backend.name: soxr
backend.quality: MQ
```

quality 候補:

- `QQ`
- `LQ`
- `MQ`
- `HQ`
- `VHQ`

要求:

- quality は startup validation で許可値に限定する。
- real-time stream では streaming API を使い、chunk ごとに one-shot resample しない。
- `soxr_delay` 相当で取得できる algorithmic delay を diagnostics / test evidence に反映する。
- `MQ` / `HQ` を主要評価対象にする。
- `VHQ` は録音 / offline / golden reference 用として評価し、realtime default にしない。

期待する性質:

- aliasing、passband ripple、stopband rejection の面で高品質。
- non-integer ratio、たとえば 44.1kHz -> 16kHz でも安定した品質を出す。
- algorithmic delay は SpeexDSP より増える可能性があるため、turn-critical path では実測後に採用判断する。

## 8. Metrics

評価指標は algorithmic delay、processing cost、audio quality、timeline stability を分けて扱う。

### 8.1 Algorithmic Delay

目的:

- resampler が signal を何 sample / 何 ms 遅らせるかを測る。

指標:

| metric | 単位 | 内容 |
| --- | --- | --- |
| `algorithmic_delay_input_samples` | samples | 入力 sample rate 基準の遅延 |
| `algorithmic_delay_output_samples` | samples | 出力 sample rate 基準の遅延 |
| `algorithmic_delay_ms` | ms | output sample rate 基準で換算した遅延 |
| `impulse_peak_offset_samples` | samples | impulse response の最大 peak 位置 |
| `group_delay_ms` | ms | sweep / impulse から推定する群遅延 |

実装側でライブラリ API から latency を取れる場合は API 値を記録する。
取れない場合、impulse response で測定する。

### 8.2 Processing Cost

目的:

- realtime chunk 処理として成立するかを見る。

指標:

| metric | 単位 | 内容 |
| --- | --- | --- |
| `processing_time_mean_ms` | ms | chunk 処理時間平均 |
| `processing_time_p95_ms` | ms | chunk 処理時間 p95 |
| `processing_time_p99_ms` | ms | chunk 処理時間 p99 |
| `realtime_factor` | ratio | audio duration / processing duration |
| `cpu_time_per_audio_second_ms` | ms | 1 秒音声あたりの CPU 時間 |

20ms chunk の pipeline では、p99 processing time が chunk duration を超えないことを最低基準にする。
ただし algorithmic delay と processing time は別物として扱う。

### 8.3 Audio Quality

目的:

- 変換後の音の差と、downsample aliasing を数値で比較する。

指標:

| metric | 単位 | 内容 |
| --- | --- | --- |
| `snr_db` | dB | high-quality reference に対する signal-to-noise ratio |
| `log_spectral_distance_db` | dB | spectrum 差分 |
| `passband_ripple_db` | dB | passband の振幅揺れ |
| `stopband_attenuation_db` | dB | stopband の抑圧量 |
| `alias_energy_db` | dB | downsample 後に折り返した成分量 |
| `peak_error` | amplitude | peak amplitude 差 |
| `rms_error` | amplitude | RMS 差 |

評価信号:

- impulse
- single-tone sine
- multi-tone sine
- logarithmic sweep
- pink noise
- 実 microphone 音声

比較軸:

- `internal_linear_resampler`
- `speexdsp` quality `3`, `5`, `6`, `8`
- `soxr` quality `MQ`, `HQ`, `VHQ`

### 8.4 Timeline Stability

目的:

- streaming chunk を通したときに timestamp / frame count が drift しないかを見る。

指標:

| metric | 単位 | 内容 |
| --- | --- | --- |
| `expected_output_frames` | frames | rational ratio から期待される累積出力 frame 数 |
| `actual_output_frames` | frames | backend が実際に出した累積 frame 数 |
| `frame_count_error_samples` | samples | 期待値との差 |
| `timestamp_drift_ns` | ns | 出力 AudioFrame timestamp の累積ずれ |
| `chunk_boundary_discontinuity` | amplitude | chunk 境界での不連続量 |

長時間 stream では、chunk ごとの丸めではなく累積 phase に基づいて出力 frame 数を決める。
1 chunk 単位の誤差ではなく、10 分程度の累積 drift を評価する。

## 9. Required Test Direction

テストは source text や docs の文字列を検査しない。

最低限必要なテスト:

- backend config validation
  - unknown backend を startup failure にする。
  - `speexdsp` quality 範囲外を拒否する。
  - `soxr` quality 許可値外を拒否する。
- backend capability validation
  - non-`FLOAT32LE` input を拒否する。
  - invalid bit depth / layout / channel / payload alignment を拒否する。
  - NaN / Inf / range 外 sample を拒否する。
- algorithm behavior
  - 48kHz -> 16kHz
  - 44.1kHz -> 16kHz
  - 16kHz -> 48kHz
  - multi-channel interleaved の channel count 維持
- streaming state behavior
  - chunk 分割しても one-shot に近い出力になる。
  - cumulative output frame count が drift しない。
  - stream state reset 後は別 stream として扱われる。
- quality metrics
  - impulse delay
  - sine / sweep の alias energy
  - reference に対する RMS / spectral error
- ROS graph behavior
  - output `AudioFrame.sample_rate == target_sample_rate`
  - `source_id` / `epoch` を維持する。
  - `stream_id` を output stream ID に更新する。
  - unsupported frame は publish しない。
- real device smoke
  - `fa_in -> fa_sample_format -> fa_resample` で microphone audio を target sample rate に変換できる。
  - diagnostics に処理時間と frame rejection count が出る。

## 10. Acceptance Criteria

この要求は、次の状態になって初めて達成とする。

1. `fa_resample` に `speexdsp` backend が追加され、config で明示選択できる。
2. `fa_resample` に `soxr` backend が追加され、config で明示選択できる。
3. `internal_linear_resampler` が暗黙 fallback ではなく、明示選択 backend として扱われる。
4. 各 backend の capability と unsupported input が明示 validation される。
5. algorithmic delay を sample / ms で説明できる evidence がある。
6. resample 後の音質差を比較できる metrics がある。
7. realtime 処理時間と cumulative drift を測る test / benchmark がある。
8. package docs が実装済み / 未実装 / 未検証を混同せず更新される。
9. representative test と real-device smoke の結果が報告される。

## 11. Node Engineer Task Input

Node Engineer は `src/processing/format/fa_resample` の実装だけを担当する。
自然言語 docs は編集しない。

変更してよい範囲:

- `src/processing/format/fa_resample/include`
- `src/processing/format/fa_resample/src`
- `src/processing/format/fa_resample/config`
- `src/processing/format/fa_resample/launch`
- `src/processing/format/fa_resample/test`
- 必要な build metadata

変更してはいけない範囲:

- `CPP_CODING_RULES.md`
- `CLAUDECODE_RULES.md`
- `PRODUCT_OWNER_ROLE.md`
- `NODE_ENGINEER_ROLE.md`
- `CLAUDECODE_DOCUMENTATION_ROLE.md`
- 親 repository
- `vlabor_ros2`
- `src/ai` の未レビュー部分変更

報告で必ず返すこと:

- 実装した backend
- backend capability
- startup failure 条件
- frame rejection 条件
- algorithmic delay の取得方法と測定値
- audio quality metrics の測定方法と結果
- processing cost の測定方法と結果
- 未検証範囲
- ClaudeCode へ渡す書類入力

## 12. ClaudeCode Documentation Task Input

ClaudeCode Documentation Writer は、Node Engineer の実装報告を受け取ってから次を更新する。

- `src/processing/format/fa_resample/docs/仕様書.md`
- `src/processing/format/fa_resample/docs/アルゴリズム詳細説明書.md`
- `src/processing/format/fa_resample/docs/テスト設計.md`
- `src/processing/format/fa_resample/docs/backends/internal_linear.md`
- `src/processing/format/fa_resample/docs/backends/speexdsp.md`
- `src/processing/format/fa_resample/docs/backends/soxr.md`

書類では、実装済み、未実装、未検証を混同しない。
backend docs には必ず capability、startup failure、frame rejection、runtime fatal、metrics、known latency を書く。

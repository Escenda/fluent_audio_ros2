# internal_threshold_gate

## 1. Backend

`internal_threshold_gate` は `fa_noise_gate` package 内の ROS 非依存 deterministic backend
である。外部 DSP library、device backend、resampler、format converter は使用しない。
ROS node は parameter、AudioFrame metadata 検証、diagnostics、publish/subscribe のみを持つ。

## 2. 入出力

- input: `FLOAT32LE` interleaved samples in normalized range `[-1.0, 1.0]`
- output: `FLOAT32LE` interleaved samples in normalized range `[-1.0, 1.0]`

## 3. Contract

backend は invalid input を補正しない。empty input、frame byte alignment error、
non-finite input、normalized range violation、invalid output は `ProcessStatus` で node に返す。
node は frame を publish せず、drop counter で可視化する。

`closed_gain_linear=0.0` は欠損値 fallback ではなく、threshold 未満の sample を明示的に
mute する設定値である。失敗した frame では output buffer と gated sample counter を commit しない。

## 4. Algorithm

```text
y = x * closed_gain_linear  when abs(x) < threshold_linear
y = x                       otherwise
```

この backend は compressor、limiter、normalize、filter、denoise ではない。

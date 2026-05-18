# internal_feedback_delay backend

## 1. 目的

`internal_feedback_delay` は `fa_reverb` package 内部で完結する deterministic reverb backend である。
外部 DSP library、model runtime、Python worker、network service には依存しない。
ROS topic、`AudioFrame` message、diagnostics は知らない。

## 2. Runtime / 依存

- C++17
- ROS2 は node layer のみ
- backend 処理自体は `std::vector<float>` delay buffer、byte列、scalar floating point 演算のみ

## 3. Input format

- `FLOAT32LE`
- `interleaved`
- normalized sample range `[-1.0, 1.0]`
- `expected.channels` ごとに独立処理

## 4. Output format

出力は `FLOAT32LE` byte列で返す。`stream_id` 更新、publish、drop counter は ROS node 側の責務である。
sample は finite かつ `[-1.0, 1.0]` を保証できる場合だけ `ProcessStatus::kOk` として返す。

## 5. Failure policy

- config 欠落または不正: node 起動失敗
- frame contract 不一致: frame drop
- non-finite input / output / state: frame drop
- normalized range 超過: frame drop
- state 不整合: frame drop

clip、clamp、zero-fill、format conversion、resample による fallback は行わない。

## 6. Diagnostics

node diagnostics は selected backend name ではなく、実際の reverb config、effective feedback gain、delay line 数、message counters を publish する。

# internal_sidechain_detector backend

`internal_sidechain_detector` は `fa_sidechain_node` から分離された ROS-free C++ backend である。ROS2 topic、`AudioFrame`、diagnostics、publisher/subscriber を知らない。

## 責務

- FLOAT32LE interleaved bytes の sample decode
- finite / normalized range validation
- RMS 計算
- threshold 判定
- active / inactive dB gain の linear 変換
- mono FLOAT32LE control payload bytes の生成
- 成功時だけ `last_rms` / `last_gain_linear` / `last_active` を更新

## 非責務

- ROS2 metadata validation
- topic / stream ID validation
- `AudioFrame` の生成
- QoS / diagnostics
- resample / normalize / clamp / limiter

## Fail Closed

不正 sample、範囲外 sample、misaligned input、表現不能な gain は `ProcessStatus` で返し、`control_data` と last state を更新しない。unknown status の message 化は `logic_error` を投げる。

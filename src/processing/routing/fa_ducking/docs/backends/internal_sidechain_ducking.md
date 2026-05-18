# internal_sidechain_ducking backend

## 概要

`internal_sidechain_ducking` は `fa_ducking_node` から分離された ROS-free C++ backend である。外部 DSP engine、device I/O、resampler、format converter は使わない。

backend は ROS2 topic、`fa_interfaces/msg/AudioFrame`、diagnostics、publisher/subscriber を知らない。

## 入力

- program FLOAT32LE byte列
- sidechain FLOAT32LE byte列
- node から渡される `now_ns`

両方とも `FLOAT32LE`、32 bit、interleaved、正規化済み sample を要求する。

## 処理

1. sidechain frame を検証し、RMS と受信時刻を recent sidechain state として保存する。
2. program frame を検証し、recent sidechain state の age と RMS から active / inactive を判定する。
3. active なら `ducking.gain_db` の linear gain、inactive なら `1.0` を target とする。
4. attack/release smoothing で current gain を更新する。
5. program sample に current gain を乗算する。

## 安全境界

- sidechain sample は output へ混ぜない。
- invalid sidechain frame は recent state を無効化する。
- invalid program frame は publish しない。
- input/output sample の正規化範囲違反は drop する。
- clamp、limiter、normalize による補正はこの backend の責務外である。

## Fail closed

- invalid sidechain input は state を commit しない。
- invalid program input/output は output buffer と current gain を commit しない。
- unknown `ProcessStatus` は `std::logic_error` で fail closed する。

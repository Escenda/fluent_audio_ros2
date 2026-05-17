# internal_sidechain_ducking backend

## 概要

`internal_sidechain_ducking` は `fa_ducking_node` 内に実装された C++ backend である。外部 DSP engine、device I/O、resampler、format converter は使わない。

## 入力

- program `fa_interfaces/msg/AudioFrame`
- sidechain `fa_interfaces/msg/AudioFrame`

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

## 将来の backend 分離

高度な graph routing、lookahead ducking、multi-band ducking が必要になった場合は、この backend を dedicated routing/dynamics backend に分離する。現在の scope では ROS2 node 内の deterministic C++ implementation を正とする。

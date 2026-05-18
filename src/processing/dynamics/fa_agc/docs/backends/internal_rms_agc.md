# internal_rms_agc backend

## 1. Backend

`internal_rms_agc` は `fa_agc` の ROS 非依存 backend である。ROS2 topic、`fa_interfaces/msg/AudioFrame`、parameter、diagnostics、publisher/subscriber は知らない。

責務は `FLOAT32LE` interleaved sample bytes を受け取り、frame RMS に基づく digital AGC を適用した sample bytes を返すことに限定する。device API、device gain API、`fa_in`、resampler、limiter、compressor、normalize backend は使わない。

## 2. Config

`InternalRmsAgcConfig` は以下を必須入力にする。

- `sample_rate`
- `channels`
- `target_rms`
- `min_gain`
- `max_gain`
- `attack_ms`
- `release_ms`

不正 config は constructor で `std::runtime_error` により fail closed する。初期 gain は `1.0` であるため、`min_gain <= 1.0 <= max_gain` を要求する。

## 3. 入力

- `std::vector<uint8_t>` の `FLOAT32LE` sample bytes
- `channels * sizeof(float)` に整列した interleaved frame
- sample は finite かつ normalized `[-1.0, 1.0]`

空入力、frame 境界に整列しない入力、非 finite sample、範囲外 sample は `ProcessStatus` で拒否する。

## 4. 処理

frame RMS を計算し、`target_rms / frame_rms` を `min_gain` から `max_gain` に制限した target gain に変換する。`frame_rms == 0.0` は valid silence として扱い、target gain は `max_gain` とする。

gain を下げる場合は attack、gain を上げる場合は release の time constant から smoothing coefficient を計算する。

## 5. 出力

成功時は output vector を AGC 後 sample bytes に置き換え、`ProcessResult` に `frame_rms`、`target_gain`、`committed_gain`、`gain_direction` を返す。

出力 sample が finite でない、または normalized range を超える場合は `ProcessStatus::kNonFiniteOutput` / `kOutOfRangeOutput` で拒否する。この場合、`current_gain`、`last_frame_rms`、`last_target_gain`、output vector は更新しない。clamp は行わない。

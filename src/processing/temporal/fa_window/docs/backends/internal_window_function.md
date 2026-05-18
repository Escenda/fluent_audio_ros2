# internal_window_function

## Backend

`internal_window_function` は `fa_window_node` から呼び出される ROS2 非依存 C++ backend である。Hann / Hamming 係数を計算し、FLOAT32LE interleaved byte列に適用する。外部 device、DSP library、resampler、format converter には依存しない。

## Contract

- 入力は `FLOAT32LE` interleaved byte列
- 係数計算に必要な frame count は `> 1`
- `strict_frame_count=true` では `window.expected_frames` のみ受け入れる
- `strict_frame_count=false` では実 frame count を係数長として使う
- invalid input sample または invalid output sample を含む frame は破棄する

## Safety Boundary

この backend は失敗を成功に見せる補正を行わない。clamp、normalize、silent fill、default fill はしない。
成功時だけ output payload を commit し、失敗時は `ProcessStatus` を返す。未知の `WindowType` / `ProcessStatus` は `std::logic_error` で fail closed する。

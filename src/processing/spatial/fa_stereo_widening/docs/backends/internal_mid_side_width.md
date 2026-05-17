# internal_mid_side_width backend

## Role

`internal_mid_side_width` は `fa_stereo_widening_node` 内部で実行する C++ 実装である。ROS2 topic、launch、QoS は node が扱い、この backend boundary は validated stereo FLOAT32LE samples に mid/side stereo width を適用する責務だけを持つ。

## Input

- FLOAT32LE little-endian byte array
- interleaved stereo
- finite normalized `[-1, 1]`
- finite `width` in `[0.0, 4.0]`

## Output

- 入力と同じ長さの FLOAT32LE little-endian byte array
- interleaved stereo
- finite normalized `[-1, 1]`

## Failure

入力または出力 sample が finite normalized `[-1, 1]` を満たさない場合、frame は drop される。clamp、normalize、推測値補完は行わない。

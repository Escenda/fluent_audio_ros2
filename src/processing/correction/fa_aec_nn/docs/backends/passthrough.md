# passthrough backend

## 目的

`passthrough` は `fa_aec_nn` の wiring 検証用 backend である。validated PCM chunk の byte payload をそのまま返し、実際の echo suppression は行わない。

default config では選択しない。利用する場合は、debug / wiring validation 用 config で `backend.name: "passthrough"` を明示する。
`enabled=false` の場合は publish せず drop する。node を実行しない場合は system config 側で node 自体を disable する。この backend は ROS-free であり、ROS2 topic、ROS message、`rclcpp` を include しない。

## 入力

- validated `AudioChunk`
- `expected_channels > 0`
- `PCM16LE/16` or `FLOAT32LE/32` explicit format pair
- interleaved layout
- non-empty PCM frame-aligned byte payload

## 出力

- 入力と同じ sample rate / channels / encoding / bit depth / layout / frame count を持つ `ProcessedAudioChunk`
- 入力と同じ byte payload

## 失敗条件

- node 側の format validation に失敗した frame
- node 側の stream-id validation に失敗した frame
- backend が empty、frame boundary 不一致、frame count 不一致の出力を返した場合
- disabled channel validation や unsupported format pair を指定した config

## 注意

この backend は NN AEC の代替ではない。未実装 model backend を `passthrough` に自動変換してはならない。

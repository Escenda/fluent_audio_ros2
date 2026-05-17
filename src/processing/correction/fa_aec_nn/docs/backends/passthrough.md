# passthrough backend

## 目的

`passthrough` は `fa_aec_nn` の wiring 検証用 backend である。validated PCM chunk の byte payload をそのまま返し、実際の echo suppression は行わない。

default config では選択しない。利用する場合は、debug / wiring validation 用 config で `backend.name: "passthrough"` を明示する。
`enabled=false` の場合は publish せず drop する。node を実行しない場合は system config 側で node 自体を disable する。この backend は ROS-free であり、ROS2 topic、ROS message、`rclcpp` を include しない。

## 入力

- validated `AudioFrame`

## 出力

- 入力と同じ `AudioFrame`

## 失敗条件

- node 側の format validation に失敗した frame

## 注意

この backend は NN AEC の代替ではない。未実装 model backend を `passthrough` に自動変換してはならない。

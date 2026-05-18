# fa_aec_nn

NN による残差エコー/非線形歪み抑圧（AEC 後段）ノードです。現状は C++ の **パススルー骨組み**です。未実装 backend 名を指定した場合は起動失敗します。

## Subscribe / Publish
- Sub: `fa_aec_nn/input`（`fa_interfaces/msg/AudioFrame`）
- Pub: `fa_aec_nn/output`（`fa_interfaces/msg/AudioFrame`）

`input_topic` / `output_topic` は ROS 搬送路です。入力 `AudioFrame.stream_id` は
`input_stream_id=audio/aec_linear/frame`、出力 `AudioFrame.stream_id` は
`output.stream_id=audio/aec/frame` として扱います。

## Run
```bash
ros2 launch fa_aec_nn fa_aec_nn.launch.py
```

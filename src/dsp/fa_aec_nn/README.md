# fa_aec_nn

NN による残差エコー/非線形歪み抑圧（AEC 後段）ノードです。現状は C++ の **パススルー骨組み**で、将来的に ONNX Runtime 推論へ置き換える前提です。

## Subscribe / Publish
- Sub: `audio/aec_linear/frame`（`fa_interfaces/msg/AudioFrame`）
- Pub: `audio/aec/frame`（`fa_interfaces/msg/AudioFrame`）

## Run
```bash
ros2 launch fa_aec_nn fa_aec_nn.launch.py
```


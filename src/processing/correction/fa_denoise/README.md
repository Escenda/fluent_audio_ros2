# fa_denoise

ノイズ抑制（NS）ノードです。

現在の既定実装は **DTLN（ONNX Runtime）** です（DeepFIRの学習済みが未公開のため、まずは公開済みモデルでベースラインを作る）。

DeepFIR 設計メモ: `docs/deepfir_ns_design_memo.md`

## Subscribe / Publish
- Sub topic: `fa_denoise/input`（`fa_interfaces/msg/AudioFrame`）
- Pub topic: `fa_denoise/output`（`fa_interfaces/msg/AudioFrame`）
- Input stream identity: `audio/resample16k/mic`
- Output stream identity: `audio/denoised/mic`

ROS topic 名は transport identity、`AudioFrame.stream_id` は audio stream identity として分離します。受信 frame の `stream_id` が `input_stream_id` と一致しない場合は drop し、publish する frame の `stream_id` は `output.stream_id` から設定します。topic 名と stream identity が一致する設定、または input/output stream identity が一致する設定は起動時に fail closed します。

## Backend
- `backend.name=dtln_onnx`（既定）:
  - 参照実装: https://github.com/breizhn/DTLN （MIT）
  - 同梱モデル: `models/model_1.onnx`, `models/model_2.onnx`
  - `dtln.model_1_path` / `dtln.model_2_path` は必須。空値から model path を推測しません。
  - 前提: 16kHz / mono / `block_len=512`, `block_shift=128`（モデルが固定）
- `backend.name=passthrough`: 入力をそのまま出力する明示 debug / wiring validation 用 backend。default/system 経路では使いません。

`enabled=false` は pass-through ではなく drop として扱います。pipeline から外す場合は system config で node 自体を disable してください。

## Build（ONNX Runtime）
`fa_denoise` は DTLN ONNX backend を前提にしますが、CI / debug 環境では ONNX Runtime なしでも package build できるように `FA_DENOISE_ONNXRUNTIME=AUTO` を既定にしています。この場合も `backend.name=dtln_onnx` は起動時に fail closed し、`passthrough` や別 backend へ暗黙に切り替えません。

- CMake 検出:
  - `ONNXRUNTIME_ROOT`（`include/` と `lib/` を含むディレクトリ）または
  - `ONNXRUNTIME_INCLUDE_DIR`, `ONNXRUNTIME_LIBRARY` を指定
- ONNX Runtime を build 時必須にする場合:
  - `-DFA_DENOISE_ONNXRUNTIME=ON`
- 例:
```bash
export ONNXRUNTIME_ROOT=$HOME/onnxruntime
colcon build --packages-select fa_denoise
```

## Run
```bash
ros2 launch fa_denoise fa_denoise.launch.py node_name:=fa_denoise config_file:=/path/to/fa_denoise.yaml
```

standalone launch では `config_file` を必ず明示してください。system config から起動する場合も、`dtln.model_1_path` / `dtln.model_2_path` を明示してください。

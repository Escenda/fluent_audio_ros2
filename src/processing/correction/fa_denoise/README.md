# fa_denoise

ノイズ抑制（NS）ノードです。

現在の既定実装は **DTLN（ONNX Runtime）** です（DeepFIRの学習済みが未公開のため、まずは公開済みモデルでベースラインを作る）。

DeepFIR 設計メモ: `docs/deepfir_ns_design_memo.md`

## Subscribe / Publish
- Sub: `audio/resample16k/mic`（`fa_interfaces/msg/AudioFrame`）
- Pub: `audio/denoise/frame`（`fa_interfaces/msg/AudioFrame`）

## Backend
- `backend=dtln_onnx`（既定）:
  - 参照実装: https://github.com/breizhn/DTLN （MIT）
  - 同梱モデル: `src/processing/correction/fa_denoise/models/model_1.onnx`, `src/processing/correction/fa_denoise/models/model_2.onnx`
  - 前提: 16kHz / mono / `block_len=512`, `block_shift=128`（モデルが固定）
- `backend=passthrough`: 入力をそのまま出力（デバッグ用）

## Build（ONNX Runtime）
`fa_denoise` は DTLN ONNX backend を前提に build します。ビルド時に ONNX Runtime（C++）が見つからない場合は configure で失敗します。

- CMake 検出:
  - `ONNXRUNTIME_ROOT`（`include/` と `lib/` を含むディレクトリ）または
  - `ONNXRUNTIME_INCLUDE_DIR`, `ONNXRUNTIME_LIBRARY` を指定
- 例:
```bash
export ONNXRUNTIME_ROOT=$HOME/onnxruntime
colcon build --packages-select fa_denoise
```

## Run
```bash
ros2 launch fa_denoise fa_denoise.launch.py
```

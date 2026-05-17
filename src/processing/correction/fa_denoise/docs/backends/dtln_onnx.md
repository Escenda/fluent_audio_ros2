# dtln_onnx backend

## 目的

`dtln_onnx` は DTLN の ONNX model を使う denoise backend である。

## 入力

- 16kHz mono float32 PCM
- `block_len=512`
- `block_shift=128`
- model 1 path
- model 2 path

## 出力

- denoise 後の float32 PCM

## runtime dependency

ONNX Runtime C++ が build 時に必要。見つからない場合、`fa_denoise` は configure で失敗する。

## 失敗条件

- model path が存在しない
- ONNX Runtime が見つからない
- input format が 16kHz mono ではない
- inference failure

失敗時に別 model や passthrough へ切り替えない。

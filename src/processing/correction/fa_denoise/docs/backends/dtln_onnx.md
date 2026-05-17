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

ONNX Runtime C++ が backend build 時に必要。CMake の `FA_DENOISE_ONNXRUNTIME` は `AUTO` / `ON` / `OFF` を取る。

- `AUTO`: ONNX Runtime が見つかる場合だけ `dtln_onnx` runtime を組み込む。見つからない場合も package は build するが、`backend.name=dtln_onnx` を選んだ node は起動時に fail closed する。
- `ON`: ONNX Runtime が見つからない場合は configure で失敗する。
- `OFF`: `dtln_onnx` runtime を組み込まない。

ONNX Runtime が無い環境で別 model、別 backend、passthrough へ暗黙に切り替えることはしない。

## 失敗条件

- model path が存在しない
- `backend.name=dtln_onnx` 選択時に ONNX Runtime support なしで build されている
- input format が 16kHz mono ではない
- inference failure

失敗時に別 model や passthrough へ切り替えない。

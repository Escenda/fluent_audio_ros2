# no_runtime_backend

## 目的

`fa_patchbay` は runtime backend を持たないことを明示する。

## 依存 runtime

なし。ROS2 node と `fa_interfaces/msg/AudioFrame` のみを使う。

## 入力 format

node parameter の `expected.*` と一致する `AudioFrame` のみを扱う。

対応 format:

- `PCM16LE` / `16` / `interleaved`
- `PCM32LE` / `32` / `interleaved`
- `FLOAT32LE` / `32` / `interleaved`

## 出力 format

入力 frame と同一である。`stream_id` のみ route output topic へ更新する。

## failure policy

backend fallback は存在しない。config が不正なら起動失敗、frame が不正なら drop する。

## diagnostics

route table と counter を `diagnostics` topic へ publish する。

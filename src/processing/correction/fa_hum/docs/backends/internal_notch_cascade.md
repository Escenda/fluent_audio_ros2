# internal_notch_cascade backend

## 1. 役割

`internal_notch_cascade` は `fa_hum` node 内部で動作する C++ backend である。外部 process や model runtime は使わず、`AudioFrame` の sample bytes に対して deterministic な biquad notch cascade を適用する。

## 2. 入力

- `FLOAT32LE`
- interleaved layout
- finite normalized samples in `[-1.0, 1.0]`
- configured sample rate / channel count と一致する frame

## 3. 出力

- 入力 metadata を保持
- `stream_id` は `output_topic`
- `data` は hum removal 後の `FLOAT32LE` bytes
- sample range は `[-1.0, 1.0]`

## 4. state

state は channel ごと、notch stage ごとに以下を保持する。

- previous input 1
- previous input 2
- previous output 1
- previous output 2

`source_id` が変わると全 state をリセットする。

## 5. fail closed policy

この backend は次の条件で処理を中止し、frame を publish しない。

- frame contract 不一致
- 入力 sample が non-finite
- 入力 sample が normalized range 外
- stage 出力が non-finite
- 最終出力が normalized range 外
- `float` 変換後の出力が non-finite または normalized range 外

出力は clamp しない。range 外 output は異常な processing result として drop する。

# internal_split_band_deesser

## 1. 役割

`internal_split_band_deesser` は `fa_deesser_node.cpp` 内に実装された内部 DSP backend
である。外部 process、device、model、network service には依存しない。

## 2. 入出力

- input: validated `FLOAT32LE` interleaved `AudioFrame`
- output: same frame identity with `stream_id` rewritten to `output_topic`

## 3. 処理

1. channel ごとの first-order low-pass state で low band を算出する。
2. `input - low` を high band とする。
3. `abs(high) >= detector.threshold` の sample だけ high band を attenuate する。
4. `low + processed_high` を出力 sample とする。

## 4. 安全境界

この backend は fallback を持たない。入力契約、filter coefficient、出力範囲のいずれかが
壊れている場合、frame を drop して diagnostics counters で可視化する。

# internal_rms_silence_removal backend

## 1. 位置づけ

`internal_rms_silence_removal` は `fa_silence_removal` node 内で実行する最小 backend である。外部推論 engine、VAD model、ASR、device I/O は持たない。

## 2. 入力

- `fa_interfaces/msg/AudioFrame`
- `FLOAT32LE`
- `bit_depth=32`
- `layout=interleaved`
- finite normalized samples in `[-1.0, 1.0]`

## 3. 出力

backend は publish 用の新しい payload を生成しない。判定結果だけを node logic に返す。

| 判定 | node の動作 |
| --- | --- |
| non-silent | 入力 frame を publish。`stream_id` のみ `output_topic` に更新 |
| silent in hangover | 入力 frame を publish。`stream_id` のみ `output_topic` に更新 |
| silent outside hangover | publish しない |
| invalid | publish しない |

## 4. 判定

全 sample の RMS を計算し、`threshold.rms` 以上を non-silent とする。

```text
rms = sqrt(sum(sample^2) / sample_count)
```

## 5. 失敗時契約

backend は invalid sample を補正しない。

- NaN / inf を 0 にしない
- out-of-range sample を clamp しない
- byte size mismatch を補正しない
- format mismatch を推定変換しない

invalid frame は node 側で warning と diagnostics counter を出し、drop する。

## 6. backend 分離の理由

この package は RMS ベースの deterministic processing であり、外部 model backend を必要としない。一方で、将来 neural VAD や ASR confidence と組み合わせる場合は、ROS2 node から推論 backend を分離し、この backend を置き換える。

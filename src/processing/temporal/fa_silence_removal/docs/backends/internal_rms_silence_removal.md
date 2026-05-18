# internal_rms_silence_removal backend

## 1. 位置づけ

`internal_rms_silence_removal` は `fa_silence_removal` node 内で実行する ROS-free C++ backend である。外部推論 engine、VAD model、ASR、device I/O は持たない。

この backend は ROS2 topic、`fa_interfaces/msg/AudioFrame`、diagnostics、publisher/subscriber を知らない。node が ROS message contract を検証した後、backend は FLOAT32LE interleaved byte列だけを受け取る。

## 2. 入力

- `std::vector<uint8_t>` の FLOAT32LE interleaved sample bytes
- `channels > 0`
- byte length は `channels * sizeof(float)` の倍数
- 各 sample は finite normalized samples in `[-1.0, 1.0]`

## 3. 出力

backend は publish 用の payload を生成しない。`ProcessStatus`、`Decision`、RMS、frame count、hangover 残量だけを node logic に返す。

| 判定 | node の動作 |
| --- | --- |
| `accepted_active` | 入力 frame を publish。`stream_id` のみ `output.stream_id` に更新 |
| `accepted_hangover` | 入力 frame を publish。`stream_id` のみ `output.stream_id` に更新 |
| silent outside hangover | publish しない |
| invalid | publish しない |

## 4. 判定

全 sample の RMS を計算し、`threshold.rms` 以上を non-silent とする。

```text
rms = sqrt(sum(sample^2) / sample_count)
```

## 5. 失敗時契約

backend は invalid input を補正しない。

- NaN / inf を 0 にしない
- out-of-range sample を clamp しない
- byte size mismatch を補正しない
- channel 数を推定しない

invalid frame は node 側で warning と diagnostics counter を出し、drop する。

backend は invalid input で `last_rms` や hangover state を更新しない。未知の `Decision` / `ProcessStatus` は `std::logic_error` で fail closed し、accepted / dropped のどちらかへ丸めない。

## 6. backend 分離の理由

この package は RMS ベースの deterministic processing であり、外部 model backend を必要としない。一方で、将来 neural VAD や ASR confidence と組み合わせる場合は、ROS2 node から推論 backend を分離し、この backend を置き換える。

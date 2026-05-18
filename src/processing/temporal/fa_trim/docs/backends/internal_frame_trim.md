# internal_frame_trim backend

`internal_frame_trim` は ROS2 に依存しない `fa_trim` の標準 backend です。

## 入力

- FLOAT32LE interleaved の byte列
- `channels > 0`
- `leading_frames` または `trailing_frames` の少なくとも一方が `> 0`

## 処理

1. 入力 byte列が空でないことを検証する。
2. `channels * sizeof(float)` で frame 境界に揃っていることを検証する。
3. 全 sample が finite かつ `[-1.0, 1.0]` であることを検証する。
4. `leading_frames` と `trailing_frames` により出力 frame が 1 frame 以上残ることを検証する。
5. 成功時のみ、trim 済み payload を output に commit する。

## 失敗

backend は失敗時に output を更新しない。`ProcessStatus` で失敗理由を返し、未知の status は `processStatusMessage()` が `std::logic_error` で fail closed する。

# no_runtime_backend

`fa_time_alignment` は外部 runtime backend を持たない。

この package の処理は `AudioFrame.header.stamp` の時間グリッド整列のみであり、device、model、network、DSP runtime を呼び出さない。

## 責務

- `AudioFrame` 契約の検証
- nearest grid timestamp の計算
- `max_adjust_ms` による drop 判定
- `header.stamp` と `stream_id` の更新
- diagnostics counters の publish

## 非責務

- audio sample decode
- audio sample edit
- resampling
- channel conversion
- device I/O
- external model invocation

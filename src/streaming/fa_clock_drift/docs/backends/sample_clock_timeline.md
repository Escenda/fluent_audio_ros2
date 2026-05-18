# sample_clock_timeline backend

## 1. 目的

`sample_clock_timeline` は `fa_clock_drift` node 内部の stateful algorithm である。外部 runtime、device、model、worker process は持たない。

## 2. Input Format

入力は validation 済み `fa_interfaces/msg/AudioFrame` である。format は node parameter の `expected.*` と完全一致している必要がある。

## 3. Output Format

出力は入力 frame の copy である。`header.stamp` は補正後 timestamp、`stream_id` は `output.stream_id` に設定する。`source_id`、`data`、`epoch` は保持する。

## 4. State

- active stream identity
- previous output timestamp
- drift estimate
- last observed drift
- reset / limit counters

## 5. Failure Policy

timestamp 計算が negative、non-finite、または `builtin_interfaces/Time` 範囲外になった場合は frame を drop し、timeline state を reset する。現在時刻や zero timestamp への代替は行わない。

## 6. Diagnostics

diagnostics は `drift_estimate_ms`、`last_observed_drift_ms`、`timeline_resets`、`correction_limited_frames` を publish する。

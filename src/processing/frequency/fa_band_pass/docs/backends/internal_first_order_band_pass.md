# internal_first_order_band_pass backend

## Role

`internal_first_order_band_pass` は `fa_band_pass_node` 内で実行される C++ 実装 backend である。外部 process、device、file、network I/O を持たない。

## Input

`fa_interfaces/msg/AudioFrame` の FLOAT32LE interleaved samples。node 境界で contract validation 済みの frame のみを処理する。

## Processing

1. accepted source の初回 sample で channel state を初期化する。
2. 各 channel に一次 high-pass recurrence を適用する。
3. high-pass output を同じ channel の一次 low-pass recurrence に入力する。
4. final output sample を FLOAT32LE interleaved `data` へ書き戻す。

## Failure Policy

backend は意味を変える fallback を行わない。invalid runtime frame、non-finite 計算結果、final output の normalized range 違反は frame drop と warning で扱う。

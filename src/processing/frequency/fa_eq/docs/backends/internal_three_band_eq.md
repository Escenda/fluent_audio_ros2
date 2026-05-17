# internal_three_band_eq backend

## Role

`internal_three_band_eq` は `fa_eq_node` 内で実行される C++ 実装 backend である。外部 process、device、file、network I/O を持たない。

## Input

`fa_interfaces/msg/AudioFrame` の FLOAT32LE interleaved samples。node 境界で contract validation 済みの frame のみを処理する。

## Processing

1. accepted source の初回 sample で channel state を初期化する。
2. 各 channel に一次 low-pass recurrence を適用して low band を得る。
3. 同じ input sample に一次 high-pass recurrence を適用して high band を得る。
4. `input - low - high` を mid band とする。
5. `low * gain_low + mid * gain_mid + high * gain_high` を final output sample として FLOAT32LE interleaved `data` へ書き戻す。

## Failure Policy

backend は意味を変える fallback を行わない。invalid runtime frame、non-finite 計算結果、final output の normalized range 違反は frame drop と warning で扱う。range 違反は clamp せず、出力しない。

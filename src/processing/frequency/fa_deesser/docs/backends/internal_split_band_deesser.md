# internal_split_band_deesser

## 1. 役割

`internal_split_band_deesser` は `fa_deesser` package 内の ROS 非依存 DSP backend
である。外部 process、device、model、network service には依存しない。
ROS node は parameter、AudioFrame metadata 検証、diagnostics、publish/subscribe のみを持つ。

## 2. 入出力

- input: validated `FLOAT32LE` interleaved sample bytes
- output: processed `FLOAT32LE` interleaved sample bytes
- status: `ProcessStatus` と attenuated sample count を返す

## 3. 処理

1. channel ごとの first-order low-pass state で low band を算出する。
2. `input - low` を high band とする。
3. `abs(high) >= detector.threshold` の sample だけ high band を attenuate する。
4. `low + processed_high` を出力 sample とする。

## 4. 安全境界

この backend は fallback を持たない。入力契約、filter coefficient、出力範囲のいずれかが
壊れている場合、`ProcessStatus` で node へ明示する。node は frame を drop して
diagnostics counters で可視化する。backend は clamp、normalize、zero fill を行わない。

## 5. 状態管理

backend は channel ごとの low-pass state を保持する。node から `reset_state=true` が
渡された場合、次 frame は全 channel state を `0.0` から処理する。
入力または出力の検証に失敗した frame では state と output を commit しない。

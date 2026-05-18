# internal_integer_bit_depth backend

## 1. 役割

`internal_integer_bit_depth` は `fa_bit_depth` node 内部で使う PCM integer bit-depth conversion backend である。
外部 process、model、device、codec library は使わない。ROS2 topic/message も知らない。

## 2. 対応形式

| input | output | 処理 |
| --- | --- | --- |
| `PCM16LE` / 16 | `PCM32LE` / 32 | 16 bit word を 32 bit word の上位 16 bit へ配置 |

layout は `interleaved` のみである。channel count は変えず、sample 順序も変えない。

## 3. 非対応

- `FLOAT32LE`
- normalized range conversion
- endian 変換
- dither
- rounding
- clipping
- lossy PCM32LE/32 to PCM16LE/16 truncation
- resample
- channel convert
- gain / limiter / normalize

非対応 config は node startup 時に fail closed で拒否する。runtime frame が入力契約に一致しない場合は warning を出して drop する。
backend は `FrameContractStatus` と `ProcessResult` で失敗理由を返す。失敗時は出力 buffer を更新しないため、呼び出し元は古い変換結果を誤って publish しない。

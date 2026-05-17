# internal_overlap_buffer

## 役割

`internal_overlap_buffer` は、`fa_chunk_overlap` node 内部の byte buffer 実装である。外部 backend process、device driver、codec、resampler ではない。

## 契約

- 入力 byte列は `FLOAT32LE` interleaved sample frames として整列済みであること。
- buffer は 1 active `source_id` の sample のみを保持する。
- `source_id` が変わった場合は buffer を clear する。
- window 未満の残り sample は保持するが publish しない。
- publish 後は `hop_samples` sample frames だけ削除する。

## 禁止事項

- zero padding
- resample
- channel conversion
- bit-depth conversion
- stale data を現在 sample として扱うこと
- invalid frame を default 値へ置換すること

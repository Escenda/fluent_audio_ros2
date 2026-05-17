# internal_fixed_chunk_buffer backend

## backend scope

この backend は外部 device や codec を持たない。内部 byte buffer のみで `AudioFrame.data` を固定 chunk に分割する。

## 入力

- `fa_interfaces/msg/AudioFrame`
- `FLOAT32LE`
- `bit_depth == 32`
- `layout == interleaved`
- `data.size()` は `channels * sizeof(float)` の整数倍

## 出力

- `data.size() == frames_per_chunk * channels * sizeof(float)`
- `stream_id == output_topic`
- format と source identity は chunk の first contributing frame 由来

## buffer upper bound

最大 byte 数は次で定義する。

```text
max_buffered_chunks * frames_per_chunk * channels * sizeof(float)
```

上限超過時は古い chunk 1 個分を削除する。これは overflow として diagnostics に記録される。

## 禁止事項

- sample value の変更
- sample format conversion
- resampling
- channel layout conversion
- partial chunk padding
- incompatible stream の merge

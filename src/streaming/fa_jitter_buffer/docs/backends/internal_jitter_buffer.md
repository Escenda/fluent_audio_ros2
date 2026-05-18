# internal_jitter_buffer backend

`fa_jitter_buffer` は外部 backend を持たない。node 内部の `std::map<uint32_t, AudioFrame>` を jitter buffer として使う。

## 責務

- valid `AudioFrame` を epoch key で保持する。
- 入力 `stream_id` が `input_stream_id` に一致する frame だけを受け入れる。
- target depth を超えた分だけ最古 epoch から publish する。
- publish 時は `stream_id` を `output.stream_id` に更新する。
- duplicate / late epoch を明示的に drop する。
- source / format contract change で buffer を reset する。

## 非責務

- audio payload の編集
- resample / format conversion
- packet loss concealment
- silence generation
- device / network I/O
- hidden retry / fallback

## 安全性

invalid frame は buffer 挿入前に drop する。欠損 frame を生成しないため、jitter は吸収できるが packet loss は隠さない。欠損補償が必要な場合は `fa_packet_loss_concealment` を別 node として pipeline に挿入する。

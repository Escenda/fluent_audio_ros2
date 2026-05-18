# fa_frame_buffer

`fa_frame_buffer` は FluentAudio の `fa_interfaces/msg/AudioFrame` を固定サイズ chunk にまとめる ROS2 streaming package です。

## 責務

- `FLOAT32LE` / 32-bit / `interleaved` の AudioFrame を購読する
- `buffering.frames_per_chunk` で指定されたフレーム数が揃ったときだけ publish する
- chunk の先頭に寄与した frame の `header`、`source_id`、format、`epoch` を出力に保持する
- 入力 `stream_id` を `input_stream_id` で検証し、出力の `stream_id` を `output.stream_id` に更新する
- partial chunk は保持し、padding は行わない
- buffer overflow 時は古い chunk 単位の transport loss として明示的に破棄し、diagnostics に count を出す

## 非責務

- device I/O
- resampling
- sample format conversion
- gain / limiter / filtering / denoise
- channel remap

## 既定設定

- input topic: `fa_frame_buffer/input`
- output topic: `fa_frame_buffer/output`
- input stream: `audio/noise_gated/mic`
- output stream: `audio/buffered/mic`
- expected: `16000Hz`, `1ch`, `FLOAT32LE`, `32bit`, `interleaved`
- chunk: `512` frames
- max buffered chunks: `4`

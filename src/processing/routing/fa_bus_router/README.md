# fa_bus_router

`fa_bus_router` は FluentAudio の `fa_interfaces/msg/AudioFrame` を、明示された複数の output topic へ複製する ROS2 processing package です。

## 責務

- 1つの `AudioFrame` input topic を購読する
- `output_topics` に列挙された topic へ、入力 frame の copy を publish する
- `stream_id` だけを publish 先 topic に更新する
- `header`、`source_id`、format metadata、`data`、`epoch` は入力 frame から保持する
- 起動時 config を fail closed で検証する
- runtime frame が期待 format と一致しない場合は drop し、warning と diagnostics に反映する

## 非責務

- mixing
- resampling
- gain
- sample format conversion
- channel conversion
- filtering / denoise
- device I/O

## 既定設定

- input: `audio/frame`
- outputs: `audio/output/frame`
- expected: `48000Hz`, `1ch`, `PCM16LE`, `16bit`, `interleaved`
- qos: reliable, depth `10`
- diagnostics: `1000ms`, `diagnostics.qos.depth=10`, `diagnostics.qos.reliable=true`

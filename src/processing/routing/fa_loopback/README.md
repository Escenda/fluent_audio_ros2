# fa_loopback

`fa_loopback` は FluentAudio の routing package である。1本の `fa_interfaces/msg/AudioFrame` stream を購読し、音声 bytes を変更せずに operator が指定した loopback topic へ publish する。

## 責務

- required `input_topic` を購読する
- required `output_topic` へ publish する
- 入力 `AudioFrame` の metadata と data を copy し、`stream_id` だけを `output_topic` に更新する
- `source_id`、`header`、format fields、`epoch`、`data` は変更しない
- invalid frame は drop し、warning と diagnostics counter に反映する

## 非責務

- resample
- clamp / normalize / gain
- decode / encode
- channel conversion
- format conversion
- device capture / playback

## self-loop guard

`loopback.require_distinct_topics` の default は `true` である。この状態で `input_topic == output_topic` の場合は startup error とする。

`false` にすると同一 topic の passthrough を許可する。これは operator が ROS2 graph 上の loopback を明示的に制御する場合だけに使う。

## default topics

- Sub: `audio/output/frame`
- Pub: `audio/loopback/frame`
- Diagnostics: `diagnostics`

## test

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider src/processing/routing/fa_loopback/test/unit -q
```

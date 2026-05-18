# fa_loopback

`fa_loopback` は FluentAudio の routing package である。1本の `fa_interfaces/msg/AudioFrame` stream を購読し、音声 bytes を変更せずに operator が指定した loopback topic へ publish する。

## 責務

- required `input_topic` を購読する
- required `output_topic` へ publish する
- required `input_stream_id` の frame だけを受け取る
- 入力 `AudioFrame` の metadata と data を copy し、`stream_id` だけを required `output.stream_id` に更新する
- `source_id`、`header`、format fields、`epoch`、`data` は変更しない
- invalid frame は drop し、warning と diagnostics counter に反映する

## 非責務

- resample
- clamp / normalize / gain
- decode / encode
- channel conversion
- format conversion
- device capture / playback

## topic と stream

`input_topic` / `output_topic` は ROS2 graph 上の配送先である。`input_stream_id` / `output.stream_id` は `AudioFrame` 内の stream identity であり、topic 名として扱わない。

`input_topic` と `output_topic` が解決後に同じ場合は startup error とする。stream ID が raw/resolved topic 名と一致する場合も startup error とする。

## launch

`fa_loopback.launch.py` は `node_name` と `config_file` を必須 launch argument とする。package 内 default config への暗黙 fallback は持たない。

`config/default.yaml` は単体デバッグ用の明示例であり、system launch では profile / system config から渡す。

## test

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider src/processing/routing/fa_loopback/test/unit -q
```

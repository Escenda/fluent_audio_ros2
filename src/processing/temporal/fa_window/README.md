# fa_window

`fa_window` は、`fa_interfaces/msg/AudioFrame` の `FLOAT32LE` interleaved stream に解析窓を適用する ROS2 processing package です。

## ノード

- package: `fa_window`
- executable: `fa_window_node`
- node name default: `fa_window`
- subscribe: `input_topic`
- publish: `output_topic`

## 入出力契約

入力は `sample_rate > 0`、`channels > 0`、`encoding=FLOAT32LE`、`bit_depth=32`、`layout=interleaved` の `AudioFrame` に限定します。`source_id` と `stream_id` は空文字を許可しません。入力 `stream_id` は `input_stream_id` と一致する必要があります。

出力は入力の `header`、`source_id`、`sample_rate`、`channels`、`bit_depth`、`encoding`、`layout`、`epoch` を保持し、`stream_id` を `output.stream_id` に更新し、`data` を window 適用済み payload に置き換えます。
窓関数計算と sample 検証は ROS2 非依存の `internal_window_function` backend が担当します。

## パラメータ

`fa_window.launch.py` は `config_file` を必須 launch argument として扱い、package 内の設定を暗黙には読み込まない。以下は `config/default.yaml` の設定例であり、runtime default ではない。

- `input_topic`: `fa_window/input`
- `output_topic`: `fa_window/output`
- `input_stream_id`: `audio/buffered/mic`
- `output.stream_id`: `audio/windowed/mic`
- `window.type`: `hann` または `hamming`
- `window.expected_frames`: `> 1`
- `window.strict_frame_count`: required bool

`strict_frame_count=true` では入力フレーム数が `expected_frames` と一致しない場合に破棄します。`false` では実際の入力フレーム数で係数を計算しますが、`frame_count > 1` は必須です。

# fa_chunk_overlap

`fa_chunk_overlap` は、FluentAudio の `fa_interfaces/msg/AudioFrame` を固定長 window に分割し、明示的な overlap を残して次の chunk を生成する ROS2 streaming node である。

この package は device I/O、resample、format conversion、padding を行わない。入力は `FLOAT32LE`、32 bit、interleaved、正規化済み float sample であることを要求する。

## Contract

- `input_topic` と `output_topic` は必須。
- `input_stream_id` と `output.stream_id` は必須で、ROS topic identity とは分ける。
- `window.frame_samples` は `> 0`。
- `window.hop_samples` は `> 0` かつ `<= window.frame_samples`。
- 入力 frame は `expected.sample_rate`、`expected.channels`、`FLOAT32LE`、`bit_depth=32`、`layout=interleaved` と一致する必要がある。
- `source_id` が変わった場合、既存 buffer を破棄して新しい source の stream として開始する。
- 不正 frame は drop し、既存の valid buffer state は変更しない。
- window が足りない場合は出力しない。zero padding は行わない。
- 出力ごとに `hop_samples` sample frames を消費し、`frame_samples - hop_samples` を overlap として残す。

## Topics

- Subscribe: `input_topic` (`fa_interfaces/msg/AudioFrame`)
- Publish: `output_topic` (`fa_interfaces/msg/AudioFrame`)
- Diagnostics: `diagnostics` (`diagnostic_msgs/msg/DiagnosticArray`)

## Parameters

設定例は `config/default.yaml` を参照。

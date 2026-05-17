# fa_overlap_add

`fa_overlap_add` は、FluentAudio の overlapped `fa_interfaces/msg/AudioFrame` chunk を hop 単位の連続 stream へ復元する ROS2 streaming node である。

入力は `FLOAT32LE`、32 bit、interleaved、正規化済み float sample の固定長 chunk に限定する。この package は device I/O、resample、format conversion、tail padding、clamp を行わない。

## Contract

- `input_topic` と `output_topic` は必須。
- `window.frame_samples` は `> 0`。
- `window.hop_samples` は `> 0` かつ `<= window.frame_samples`。
- `window.type` は `rectangular` または `hann`。
- `overlap.max_buffered_chunks` は `> 0`。
- 入力 frame は `expected.sample_rate`、`expected.channels`、`FLOAT32LE`、`bit_depth=32`、`layout=interleaved` と一致する必要がある。
- 入力 `data` は `window.frame_samples * channels * sizeof(float)` bytes ちょうどである必要がある。
- 不正 frame は drop し、既存の valid overlap-add state は変更しない。
- `source_id`、format、future input `epoch` gap が出た場合は state を reset して新 chunk から開始する。
- duplicate / regressing input `epoch` は stale audio replay を避けるため drop する。
- 出力は hop-sized `AudioFrame` とし、`source_id` を保持し、`stream_id` を `output_topic` に更新する。

## Topics

- Subscribe: `input_topic` (`fa_interfaces/msg/AudioFrame`)
- Publish: `output_topic` (`fa_interfaces/msg/AudioFrame`)
- Diagnostics: `diagnostics` (`diagnostic_msgs/msg/DiagnosticArray`)

## Parameters

設定例は `config/default.yaml` を参照。詳細な仕様、アルゴリズム、テスト対応は `docs/` 配下に分ける。

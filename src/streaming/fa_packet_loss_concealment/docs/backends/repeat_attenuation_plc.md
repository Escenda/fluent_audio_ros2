# repeat_attenuation_plc backend

## 1. 目的

`repeat_attenuation_plc` は、`fa_packet_loss_concealment` の内部 backend である。欠落した短い epoch 区間だけ、直前の有効 frame を減衰して繰り返す。

## 2. Runtime / dependencies

- 外部 process なし
- device I/O なし
- model artifact なし
- C++17 standard library のみ
- ROS2 topic を扱うのは node 側のみ

## 3. Input format

- `fa_interfaces/msg/AudioFrame`
- `FLOAT32LE`
- 32bit
- `interleaved`
- finite normalized sample

## 4. Output format

synthetic frame は previous frame の format を維持し、`stream_id` だけ `output.stream_id` に変更する。`epoch` は欠落 epoch と一致させる。ROS topic は node 側の transport 設定であり、backend の stream identity には使わない。

## 5. Failure policy

- 前回 frame がない場合は合成しない。
- gap が `plc.max_gap_frames` を超える場合は合成しない。
- duplicate / regression epoch は drop する。
- timestamp advance が ROS2 time 範囲を超える場合は合成しない。

## 6. Diagnostics

- `backend.name=repeat_attenuation_plc`
- `concealed_frames`
- `gap_resets`
- `duplicate_drops`

## 7. Test fixture

unit contract test は `test/unit/test_fa_packet_loss_concealment_audio_frame_contract.py` に置く。numeric PCM fixture は `test/fixtures` に追加可能だが、巨大な録音データは置かない。

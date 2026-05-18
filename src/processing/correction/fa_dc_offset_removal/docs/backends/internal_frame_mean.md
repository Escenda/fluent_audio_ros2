# internal_frame_mean backend

`internal_frame_mean` は ROS 非依存の frame-local DC offset removal backend である。ROS2 topic、`fa_interfaces/msg/AudioFrame`、device、resampler、format converter は知らない。

## 入力契約

- `std::vector<uint8_t>` の `FLOAT32LE` sample bytes
- `channels > 0`
- `data.size() > 0`
- `data.size() % (channels * sizeof(float)) == 0`
- 全 sample が finite

## 出力契約

- 入力と同じ sample 数の `FLOAT32LE` byte列
- channel ごとの frame 内平均を差し引いた sample
- 非有限値が発生した場合は出力せず `ProcessStatus` を返す

拒否時は output vector を更新しない。warning、drop counter、publish 抑止は ROS node 側の責務である。

## 非責務

- DC offset の長期推定
- high-pass filtering
- notch filtering
- gain / limiter / noise gate
- sample format conversion
- resampling
- device I/O

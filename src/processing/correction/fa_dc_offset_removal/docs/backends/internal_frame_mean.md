# internal_frame_mean backend

`internal_frame_mean` は `fa_dc_offset_removal` に内包される frame-local DC offset removal backend である。ROS2 topic、device、resampler、format converter は知らない。

## 入力契約

- `FLOAT32LE`
- `bit_depth == 32`
- `layout == interleaved`
- `channels > 0`
- `data.size() > 0`
- `data.size() % (channels * sizeof(float)) == 0`
- 全 sample が finite

## 出力契約

- 入力と同じ sample 数の `FLOAT32LE` byte列
- channel ごとの frame 内平均を差し引いた sample
- 非有限値が発生した場合は出力せず失敗を返す

## 非責務

- DC offset の長期推定
- high-pass filtering
- notch filtering
- gain / limiter / noise gate
- sample format conversion
- resampling
- device I/O

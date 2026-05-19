# static_delay_sum backend

## 目的

`static_delay_sum` は、明示された channel weight を使って multi-channel FLOAT32LE interleaved PCM を mono PCM へ変換する初期 backend である。

## 依存 Runtime

外部 DSP library、model runtime、worker process は使わない。C++ 標準 library のみで実装する。

## Input Format

- encoding: `FLOAT32LE`
- bit depth: `32`
- layout: `interleaved`
- channels: `expected.channels`
- sample range: finite `[-1.0, 1.0]`

## Output Format

- encoding: `FLOAT32LE`
- bit depth: `32`
- layout: `interleaved`
- channels: `1`
- sample range: finite `[-1.0, 1.0]`

## Weight Contract

- `beamforming.weights` は double array
- 長さは `expected.channels` と一致
- 全要素 finite
- 絶対値和が 0 より大きい
- node は weight を normalize しない
- node は weight を推定しない
- node は equal-weight fallback を持たない

## Failure Policy

config 不正は起動失敗。runtime frame 不正は warning と counter 更新のうえ frame drop。clamp、zero fill、normalize、default weight 補完は行わない。

## Diagnostics

`beamforming.weights`、`beamforming.weights_sum_abs`、expected contract、output contract、frame counter を publish する。

## Test Fixture

初期 contract test は config validation と backend public API の capability contract を検証する。将来の数値 fixture は短い synthetic PCM を `test/fixtures` に追加する。

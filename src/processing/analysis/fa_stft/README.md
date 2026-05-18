# fa_stft

`fa_stft` is a non-AI `processing/analysis` package that converts canonical
`AudioFrame` PCM samples into STFT feature frames.

It does not perform resampling, channel conversion, sample format conversion,
VAD, ASR, or model inference. Input audio must already be mono `FLOAT32LE` /
32-bit / interleaved / normalized `[-1.0, 1.0]` and match the configured stream
contract.

Details are in `docs/仕様書.md`, `docs/アルゴリズム詳細説明書.md`,
`docs/テスト設計.md`, and `docs/backends/internal_stft.md`.

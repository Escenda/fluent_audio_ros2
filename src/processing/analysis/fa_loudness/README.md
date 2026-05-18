# fa_loudness

`fa_loudness` is a non-AI `processing/analysis` package that measures frame
RMS, peak, dBFS, and crest factor from canonical `AudioFrame` PCM samples.

It does not normalize, limit, compress, resample, convert sample format, or
perform LUFS/K-weighted metering. Those belong to explicit future backends or
dedicated processing nodes.

Details are in `docs/仕様書.md`, `docs/アルゴリズム詳細説明書.md`,
`docs/テスト設計.md`, and `docs/backends/internal_frame_meter.md`.

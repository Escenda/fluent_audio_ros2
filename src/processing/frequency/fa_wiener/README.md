# fa_wiener

`fa_wiener` は Wiener-style noise suppression を audio frames に適用する frequency processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/stft_wiener_filter.md`

この node は周波数領域の noise suppression であり、device I/O、resample、gain、VAD、ASR は扱いません。

# fa_spectral_subtraction

`fa_spectral_subtraction` は noise spectrum を推定し、incoming audio frames から差し引く frequency processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/stft_noise_profile.md`

この node は周波数領域の noise reduction であり、device I/O、resample、gain、VAD、ASR は扱いません。

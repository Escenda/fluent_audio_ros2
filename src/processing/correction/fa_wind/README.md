# fa_wind

`fa_wind` は input audio stream の wind noise を低減する correction processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/wind_noise_reduction.md`

この node は wind noise reduction のみを扱い、high-pass filter、noise gate、beamforming、source adapter は扱いません。

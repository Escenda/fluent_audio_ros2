# fa_ambisonics

`fa_ambisonics` は ambisonic sound field representation と transform を扱う spatial processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/ambisonic_transform.md`

この node は sound field channel transform のみを扱い、device I/O、binaural rendering、beamforming、source separation は扱いません。

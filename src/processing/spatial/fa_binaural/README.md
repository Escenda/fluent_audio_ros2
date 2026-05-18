# fa_binaural

`fa_binaural` は headphone-oriented binaural output へ rendering する spatial processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/hrtf_renderer.md`

この node は HRTF rendering のみを扱い、speaker output、fa_out sink、ambisonic transform、source separation は扱いません。

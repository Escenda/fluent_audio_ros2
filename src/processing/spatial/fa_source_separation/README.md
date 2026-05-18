# fa_source_separation

`fa_source_separation` は audio mixture から speech / music / noise sources を分離する spatial processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_separator.md`

この node は source separation のみを扱い、beamforming、speaker embedding、ASR、routing mixer は扱いません。

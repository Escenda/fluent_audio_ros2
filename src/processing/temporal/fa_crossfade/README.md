# fa_crossfade

`fa_crossfade` は audio segments を明示的な overlap と fade curve で接続する temporal processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/internal_crossfade.md`

この node は複数 stream / segment の時間方向接続だけを扱い、device I/O、resample、gain normalize、routing mixer は扱いません。

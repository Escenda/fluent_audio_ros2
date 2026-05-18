# fa_super_resolution

`fa_super_resolution` は lower-resolution input audio から audio bandwidth を復元・拡張する generation processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_audio_super_resolution.md`

この node は bandwidth extension のみを扱い、resample、EQ、denoise、speaker output は扱いません。

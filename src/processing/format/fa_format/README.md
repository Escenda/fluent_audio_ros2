# fa_format

`fa_format` は format conversion stages を明示的に組み合わせる wrapper / pipeline node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/explicit_format_pipeline.md`

この node は resample、channel、bit-depth、codec、sample format 変更を `fa_in` / `fa_out` に隠さず、設定された順序で専用 node 群を include / compose する境界です。

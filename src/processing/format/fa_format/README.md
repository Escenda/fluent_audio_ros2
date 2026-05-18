# fa_format

`fa_format` は format conversion stages を明示的に組み合わせる wrapper / pipeline node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

Current `fluent_audio_system` profiles must not enable `package: fa_format`.
Until `fa_format` becomes a declared ROS 2 package, format pipelines are written
with the concrete leaf packages such as `fa_decode`, `fa_sample_format`,
`fa_resample`, `fa_bit_depth`, `fa_channel_convert`, `fa_interleave`, and
`fa_encode`.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/explicit_format_pipeline.md`

この node は resample、channel、bit-depth、codec、sample format 変更を `fa_in` / `fa_out` に隠さず、設定された順序で専用 node 群を include / compose する境界です。

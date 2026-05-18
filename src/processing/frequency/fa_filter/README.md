# fa_filter

`fa_filter` は frequency processing stages を明示的に組み合わせる wrapper / pipeline node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/explicit_filter_pipeline.md`

この node は high-pass、low-pass、band-pass、notch、EQ、de-esser、spectral subtraction、Wiener filtering などを単一の hidden filter に畳み込まず、専用 stage として接続する境界です。
ROS topic は搬送路として扱い、各 stage の `AudioFrame.stream_id` は
`stage.input_stream_id` / `stage.output.stream_id` で明示します。

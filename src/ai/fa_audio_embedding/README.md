# fa_audio_embedding

`fa_audio_embedding` は audio embedding を生成・提供する AI node の設計ディレクトリです。

現時点では ROS 2 package ではありません。`package.xml`、launch、config、node 実装を持たないため、実装済みとして扱いません。

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_audio_embedding.md`

backend は ROS2 topic/message を知らない external worker / process / container 境界に分けます。

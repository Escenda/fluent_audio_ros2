# fa_audio_embedding

`fa_audio_embedding` は canonical `AudioFrame` から audio embedding を生成して
`AudioEmbeddingFrame` として publish する ROS 2 package です。

model runtime は ROS 2 node に直接混ぜず、`backends/` 配下の external worker / process /
container 境界として扱います。backend は ROS 2 topic/message を知りません。

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_audio_embedding.md`

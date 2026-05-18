# fa_neural_codec

`fa_neural_codec` は waveform audio と neural codec representation の相互変換を扱う generation processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_neural_codec.md`

この node は neural codec encode/decode のみを扱い、transport codec、Opus/FLAC encode、TTS、ASR は扱いません。

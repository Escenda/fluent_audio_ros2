# fa_speech_separation

`fa_speech_separation` は複数話者の speech sources を明示 output stream へ分離する generation processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_speech_separator.md`

この node は speech source separation のみを扱い、diarization、speaker embedding、ASR、routing mixer は扱いません。

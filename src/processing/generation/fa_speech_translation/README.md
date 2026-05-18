# fa_speech_translation

`fa_speech_translation` は speech in one language を別言語 speech へ変換する generation processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_speech_translation.md`

この node は speech-to-speech translation のみを扱い、ASR transcript generation、TTS standalone synthesis、dialogue policy は扱いません。

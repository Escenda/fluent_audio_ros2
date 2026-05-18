# fa_voice_conversion

`fa_voice_conversion` は explicit data-plane contract を保ちながら voice characteristics を変換する generation processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_voice_conversion.md`

この node は voice conversion のみを扱い、TTS、speaker embedding generation、ASR、dialogue policy は扱いません。

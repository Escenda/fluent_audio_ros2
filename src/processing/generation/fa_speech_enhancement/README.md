# fa_speech_enhancement

`fa_speech_enhancement` は noisy input audio から enhanced speech audio を生成する generation processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_speech_enhancer.md`

この node は speech enhancement のみを扱い、VAD、ASR、dereverb、AEC、source adapter は扱いません。

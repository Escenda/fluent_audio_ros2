# fa_neural_vocoder

`fa_neural_vocoder` は mel spectrogram などの acoustic features から waveform audio を復元する generation processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_neural_vocoder.md`

この node は feature-to-waveform generation のみを扱い、TTS text frontend、ASR、resample、speaker output は扱いません。

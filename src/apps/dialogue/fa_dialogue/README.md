# fa_dialogue

`fa_dialogue` は wake word、ASR、turn detection、TTS、外部 reasoning service をつなぐ対話 orchestration app の設計ディレクトリです。

現時点では ROS 2 package ではありません。`package.xml`、launch、config、node 実装を持たないため、実装済みとして扱いません。

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_dialogue_service.md`

## Boundary

この package は dialogue state と action proposal を扱います。ASR、KWS、TTS waveform generation、低レベル audio frame processing、robot command の最終安全判断は扱いません。

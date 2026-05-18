# fa_music_source_separation

`fa_music_source_separation` は音楽 mixture から vocals / drums / bass / accompaniment などを分離する generation processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_music_separator.md`

この node は music source separation のみを扱い、routing mixer、speaker diarization、ASR、device I/O は扱いません。

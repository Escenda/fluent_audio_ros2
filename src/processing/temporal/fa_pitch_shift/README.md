# fa_pitch_shift

`fa_pitch_shift` は duration を保ったまま pitch を変更する temporal processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/phase_vocoder_pitch_shift.md`

この node は pitch transformation のみを扱い、resample、speaker output、ASR feature extraction、voice conversion は扱いません。

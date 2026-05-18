# fa_debreath

`fa_debreath` は breath noise を抑制する correction processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/breath_suppression.md`

この node は breath suppression のみを扱い、source/sink adapter、VAD、ASR、gain normalize は扱いません。

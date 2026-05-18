# fa_declip

`fa_declip` は clipped audio waveform を補正する correction processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/waveform_declip.md`

この node は clipping 補正のみを扱い、limiter、gain、normalize、source/sink adapter は扱いません。

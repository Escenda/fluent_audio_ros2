# fa_time_stretch

`fa_time_stretch` は pitch を保ったまま duration を変更する temporal processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/phase_vocoder_time_stretch.md`

この node は duration transformation のみを扱い、resample、latency compensation、jitter buffer、pitch shift は扱いません。

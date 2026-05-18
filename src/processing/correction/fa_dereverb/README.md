# fa_dereverb

`fa_dereverb` は room reverberation を低減する correction processing node の設計ディレクトリです。

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
its specification, algorithm notes, backend documentation, launch contract, and
tests.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/reverb_suppression.md`

この node は dereverberation のみを扱い、AEC、beamforming、source separation、ASR は扱いません。

# fa_file_in

`fa_file_in` is the design-map directory for a future standalone file source
adapter.

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
a complete launch contract, parameter contract, backend contract, and executable
tests.

Current FluentAudio profiles must not enable `package: fa_file_in`. Raw PCM file
input is currently handled by `fa_in` through its explicit `pcm_file_reader`
source backend.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/pcm_file_reader_adapter.md`

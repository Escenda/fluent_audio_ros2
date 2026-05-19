# fa_file_out

`fa_file_out` is the design-map directory for a future standalone file sink
adapter.

This is not a ROS 2 package yet. Do not add `package.xml` until the package has
a complete launch contract, parameter contract, backend contract, and executable
tests.

Current FluentAudio profiles must not enable `package: fa_file_out`. Raw PCM file
output is currently handled by `fa_out` through its explicit `pcm_file_writer`
sink backend.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/pcm_file_writer_adapter.md`

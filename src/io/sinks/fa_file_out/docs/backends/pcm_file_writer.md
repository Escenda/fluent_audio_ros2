# pcm_file_writer backend

`pcm_file_writer` は incoming `AudioFrame` payload を明示 file target へ raw PCM として write する backend contract である。

## Required Config

- `backend.name`
- `file.path`
- `expected.sample_rate`
- `expected.channels`
- `expected.encoding`
- `expected.bit_depth`
- `expected.layout`
- `overwrite.enabled`

## Forbidden

- hidden codec encode
- hidden resample
- hidden channel conversion
- hidden gain / normalize / limiter
- output path guessing

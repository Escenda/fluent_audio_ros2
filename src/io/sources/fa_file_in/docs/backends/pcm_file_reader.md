# pcm_file_reader backend

`pcm_file_reader` は raw PCM file を source adapter として読む backend contract である。

## Required Config

- `backend.name`
- `file.path`
- `expected.sample_rate`
- `expected.channels`
- `expected.encoding`
- `expected.bit_depth`
- `expected.layout`

## Forbidden

- hidden decode
- hidden resample
- hidden gain / normalize
- missing file fallback


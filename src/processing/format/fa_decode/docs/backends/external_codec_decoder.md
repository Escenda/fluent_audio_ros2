# external_codec_decoder backend

`external_codec_decoder` は ffmpeg / libav / gstreamer / custom process などを外部 decode backend として扱うための契約である。

## Required Config

- `backend.name`
- `backend.command` または `backend.library`
- `codec`
- `output.sample_rate`
- `output.channels`
- `output.encoding`
- `output.bit_depth`
- `output.layout`

## Contract

backend は encoded payload を受け取り、decoded samples と actual format metadata を返す。
actual format metadata が config と一致しない場合、node は publish しない。

## Forbidden

- decoder 内での hidden resample
- decoder 内での hidden channel conversion
- decoder 内での hidden gain / normalize
- decode failure 時の zero-filled frame
- stale frame reuse


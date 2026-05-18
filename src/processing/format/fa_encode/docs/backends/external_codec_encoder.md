# external_codec_encoder backend

`external_codec_encoder` は ffmpeg / libav / gstreamer / custom process などを外部 encode backend として扱うための契約である。

## Required Config

- `backend.name`
- `backend.command` または `backend.library`
- `codec`
- `input.sample_rate`
- `input.channels`
- `input.encoding`
- `input.bit_depth`
- `input.layout`
- `output.container`
- `output.bitrate` または codec 固有 quality setting

## Contract

backend は PCM samples と input format metadata を受け取り、encoded payload と actual codec metadata を返す。
actual codec metadata が config と一致しない場合、node は publish しない。

## Forbidden

- encoder 内での hidden resample
- encoder 内での hidden channel conversion
- encoder 内での hidden gain / normalize
- encode failure 時の silence payload
- stale packet reuse


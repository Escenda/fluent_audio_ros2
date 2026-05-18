# external_codec_decoder backend

`external_codec_decoder` は ffmpeg / libav / gstreamer / custom process などを外部 decode backend として扱うための契約である。

## Required Config

- `backend.name`
- `backend.command.executable`
- `backend.command.timeout_ms`
- `backend.command.max_output_bytes`
- `input.codec`
- `input.container`
- `input.payload_format`
- `input.sample_rate`
- `input.channels`
- `output.sample_rate`
- `output.channels`
- `output.encoding`
- `output.bit_depth`
- `output.layout`

## Contract

backend は encoded payload を受け取り、decoded samples と startup config の output contract を返す。
stdout は byte payload のみとして扱い、sample rate/channel count/encoding/bit depth/layout を stdout から推定しない。
output contract が config と一致しない場合、node は publish しない。
command は shell 経由では実行せず、executable と arguments を分離して起動する。
`backend.command.arguments` は string list として必ず指定する。追加引数が不要な場合も `[]` を明示する。
引数なしの場合は ROS2 YAML の空配列型解決に依存せず、この parameter 自体を省略する。

## Forbidden

- decoder 内での hidden resample
- decoder 内での hidden channel conversion
- decoder 内での hidden gain / normalize
- decode failure 時の zero-filled frame
- stale frame reuse
- shell string の暗黙解釈
- command timeout 時の処理継続
- stdout payload からの PCM metadata 推測

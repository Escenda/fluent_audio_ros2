# external_codec_encoder backend

`external_codec_encoder` は ffmpeg / libav / gstreamer / custom process などを外部 encode backend として扱うための契約である。

## Required Config

- `backend.name`
- `backend.command.executable`
- `backend.command.timeout_ms`
- `backend.command.max_output_bytes`
- `codec`
- `input.sample_rate`
- `input.channels`
- `input.encoding`
- `input.bit_depth`
- `input.layout`
- `output.container`
- `output.bitrate` または codec 固有 quality setting

## Contract

backend は PCM samples と input format metadata を受け取り、encoded payload と startup config の output contract を返す。
stdout は byte payload のみとして扱い、codec/container/payload_format を stdout から推定しない。
output contract が config と一致しない場合、node は publish しない。
command は shell 経由では実行せず、executable と arguments を分離して起動する。
追加引数が必要な backend だけ `backend.command.arguments` を string list として指定する。
引数なしの場合は ROS2 YAML の空配列型解決に依存せず、この parameter 自体を省略する。

## Forbidden

- encoder 内での hidden resample
- encoder 内での hidden channel conversion
- encoder 内での hidden gain / normalize
- encode failure 時の silence payload
- stale packet reuse
- shell string の暗黙解釈
- command timeout 時の処理継続
- stdout payload からの codec metadata 推測

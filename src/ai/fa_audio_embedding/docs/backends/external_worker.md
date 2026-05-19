# external_worker backend

`external_worker` は audio embedding model を external worker / process / container として扱う backend contract である。
backend は ROS2 topic/message を知らず、node が検証済みの canonical audio payload だけを受け取る。

## Required Config

- `backend.name: external_worker`
- `backend.command`
- `backend.args`
- `backend.model_id`
- `backend.model_path`、file artifact が必要な場合のみ
- `embedding.dimension`
- `backend.payload_encoding`
- supported sample rates
- supported window
- execution mode: batch embedding または streaming embedding
- output schema: whitespace-separated `float32` vector

`backend.args` は `{audio}`、`{model_id}`、`{sample_rate}`、`{dimension}` を必ず含む。
`backend.model_path` は worker が file artifact を必要とする場合だけ指定する。指定した場合は path が存在し、読み取り可能で、`backend.args` に `{model_path}` を含めなければならない。
`{source_id}`、`{stream_id}` は必要な backend だけが使う。

worker input は raw `float32le` mono payload とし、worker output は stdout 上の whitespace-separated `float32` vector とする。payload は node が `AudioFrame` から検証して書き出したものに限定する。

## Supported AudioFrame Input

`external_worker` backend が受け取れる入力は次の `AudioFrame` contract に一致するものだけである。

- `encoding=FLOAT32LE`
- `bit_depth=32`
- `channels=1`
- `layout=interleaved`
- configured sample rate
- configured window
- finite normalized samples in `[-1.0, 1.0]`

`PCM16`、`PCM32`、stereo、planar、sample-rate mismatch、window mismatch、non-finite sample、range violation は backend 内で補正しない。node 側で frame reject し、backend に渡った場合も capability mismatch として fail closed にする。

## Model Capability

worker capability は次の fields を持つ。

- model id: `backend.model_id`
- model path: `backend.model_path`、必要な場合のみ
- provider: `external_worker`
- worker command: `backend.command`
- supported sample rates
- supported window
- execution mode: batch embedding または streaming embedding
- output schema: whitespace-separated `float32` vector
- embedding dimension

## Forbidden

- ROS2 topic/message dependency inside backend
- zero vector fallback
- stale embedding reuse
- missing model fallback

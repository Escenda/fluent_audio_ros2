# external_worker backend

`external_worker` は audio embedding model を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name: external_worker`
- `backend.command`
- `backend.args`
- `backend.model_id`
- `embedding.dimension`
- `backend.payload_encoding`

`backend.args` は `{audio}`、`{model_id}`、`{sample_rate}`、`{dimension}` を必ず含む。
`backend.model_path` は worker が file artifact を必要とする場合だけ指定する。指定した場合は path が存在し、読み取り可能で、`backend.args` に `{model_path}` を含めなければならない。
`{source_id}`、`{stream_id}` は必要な backend だけが使う。

worker input は raw `float32le` mono payload とし、worker output は stdout 上の whitespace-separated `float32` vector とする。

## Forbidden

- ROS2 topic/message dependency inside backend
- zero vector fallback
- stale embedding reuse
- missing model fallback

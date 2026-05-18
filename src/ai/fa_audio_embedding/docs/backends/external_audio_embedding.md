# external_audio_embedding backend

`external_audio_embedding` は audio embedding model を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name`
- `backend.command`
- `backend.args`
- `backend.model_id`
- `embedding.dimension`
- `payload_encoding`

`backend.args` は `{audio}`、`{model_id}`、`{sample_rate}`、`{dimension}` を必ず含む。
`{model_path}`、`{source_id}`、`{stream_id}` は必要な backend だけが使う。

worker input は raw `float32le` mono payload とし、worker output は stdout 上の whitespace-separated `float32` vector とする。

## Forbidden

- ROS2 topic/message dependency inside backend
- zero vector fallback
- stale embedding reuse
- missing model fallback

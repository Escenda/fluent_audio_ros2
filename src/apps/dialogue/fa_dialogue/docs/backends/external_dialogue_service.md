# external_dialogue_service backend

この文書は後続 slice のための未実装 contract である。今回の `fa_dialogue_node` は external dialogue backend を起動・呼び出し・設定解決しない。

`external_dialogue_service` は dialogue reasoning を external worker / service / process として扱う backend contract である。ROS 2 node は backend の model runtime を import せず、typed request / response 境界だけを持つ。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_id`
- `timeout_ms`
- `request_schema`
- `response_schema`

## Required Response Fields

- `session_id`
- `turn_id`
- `response_text`
- `action_proposal`
- `requires_safety_policy`
- `reason`

## Forbidden

- ROS2 topic/message dependency inside backend
- missing backend fallback
- default model fallback
- canned response fallback
- direct robot actuation
- final safety accept / reject

## Failure Policy

timeout、unavailable、schema mismatch、session mismatch、turn mismatch は fail closed とする。dialogue node は action proposal を publish せず、error reason を明示する。

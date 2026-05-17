# Repository Backend Boundary

FluentAudio repository root は runtime backend を持ちません。Backend 実装は各 node package の `docs/backends/` と `src` / package module 配下の `backends/` に置きます。

この文書は repository 横断の backend 境界だけを定義します。

## 共通契約

- backend は ROS2 topic / message / launch を知らない。
- backend selection は `backend.name` で明示する。
- unknown backend は起動失敗にする。
- missing model、missing executable、missing endpoint、missing credential は起動失敗にする。
- OpenAI や外部 service は明示 backend であり、local backend の fallback ではない。
- Python version / venv が異なる engine は external process / worker / container 境界へ分離する。


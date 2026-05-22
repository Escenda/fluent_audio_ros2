# fa_audio_mcp

`fa_audio_mcp` は FluentAudio timeline services を MCP tools として公開する ROS2 `ament_python` package です。


## 目的

- Agent / MCP client から FluentAudio の音声 window export、archive、文字起こし service を呼び出せるようにする。
- Agent-facing な `audio_scope` を、export 用 scope mapping、archive 用 scope mapping、transcribe 用 scope mapping で別々に解決する。
- omitted / null / blank の `audio_scope` を tool ごとの explicit default scope key で解決する。
- ROS2 service の `success=false`、service timeout、service unavailable を tool error として返す。

## 非責務

- LLM 判断、tool 選択、会話 state machine
- 自然言語 time range 解決
- marker / turn / action id からの range 解決
- audio decode / resample / format conversion
- unsupported scope の default fallback
- transcribe scope default の暗黙有効化
- ROS service error の握りつぶし

## Tool Boundary

| Tool | ROS2 service | 主な入力 | scope mapping |
| --- | --- | --- | --- |
| `export_audio_window` | `ExportAudioWindow` | `time_range`, `audio_scope`, optional format fields | `FLUENT_AUDIO_EXPORT_SCOPE_*` |
| `archive_audio_window` | `ArchiveAudioWindow` | `time_range`, `audio_scope`, `reason`, `related_artifact_ids`, optional format fields | `FLUENT_AUDIO_ARCHIVE_SCOPE_*` |

`time_range` は `<start_unix_ns>..<end_unix_ns>`、または `now-10s..now` / `now-1500ms..now-500ms` / `now-2m..now-1m` のような `now[-duration]..now[-duration]` を受け付けます。`now` は `fa_audio_mcp_server` の ROS node clock で一度だけ解決され、下流 service には numeric range だけを渡します。marker、turn id、action id、自然言語表現はここでは解決しません。

`export_audio_window` / `archive_audio_window` の format 省略時は adapter 側で `pcm_s16le` / `wav` / `audio/wav` を明示値として request に入れます。これは audio clip request の default contract であり、音声 decode や hidden conversion ではありません。


## Environment Variables

| 環境変数 | default | 内容 |
| --- | --- | --- |
| `FLUENT_AUDIO_MCP_TRANSPORT` | `stdio` | `stdio`, `sse`, `streamable-http` のいずれか |
| `FLUENT_AUDIO_MCP_HOST` | `0.0.0.0` | `sse` / `streamable-http` 用 bind host |
| `FLUENT_AUDIO_MCP_PORT` | `9110` | `sse` / `streamable-http` 用 port。正の整数 |
| `FLUENT_AUDIO_MCP_SERVICE_TIMEOUT_SEC` | `10.0` | ROS service wait / response timeout。正の数 |
| `FLUENT_AUDIO_EXPORT_AUDIO_WINDOW_SERVICE` | `export_audio_window` | export 用 ROS service 名 |
| `FLUENT_AUDIO_ARCHIVE_AUDIO_WINDOW_SERVICE` | `archive_audio_window` | archive 用 ROS service 名 |
| `FLUENT_AUDIO_EXPORT_SCOPE_MIC` | `mic` | export tool の `mic` scope 解決先 |
| `FLUENT_AUDIO_EXPORT_SCOPE_SYSTEM` | 未設定 | export tool の `system` scope 解決先 |
| `FLUENT_AUDIO_EXPORT_SCOPE_MIXED` | 未設定 | export tool の `mixed` scope 解決先 |
| `FLUENT_AUDIO_EXPORT_DEFAULT_SCOPE` | `mic` | export tool の null / blank `audio_scope` 解決 key |
| `FLUENT_AUDIO_ARCHIVE_SCOPE_MIC` | `mic` | archive tool の `mic` scope 解決先 |
| `FLUENT_AUDIO_ARCHIVE_SCOPE_SYSTEM` | 未設定 | archive tool の `system` scope 解決先 |
| `FLUENT_AUDIO_ARCHIVE_SCOPE_MIXED` | 未設定 | archive tool の `mixed` scope 解決先 |
| `FLUENT_AUDIO_ARCHIVE_DEFAULT_SCOPE` | `mic` | archive tool の null / blank `audio_scope` 解決 key |

未設定 scope は unsupported scope として fail closed します。`export`、`archive`、`transcribe` は別 mapping なので、export / archive の `mic` default が transcribe に流用されることはありません。


```bash
export FLUENT_AUDIO_EXPORT_SCOPE_MIC=mic
export FLUENT_AUDIO_ARCHIVE_SCOPE_MIC=mic
```

`fluent_audio_system/config/profiles/so101_agent_audio_tools.yaml` は、この SO101 用 mapping と `streamable-http` transport を system profile 側の node environment として明示します。

## Run

ROS2 workspace を source 済みの shell で起動します。

```bash
ros2 run fa_audio_mcp fa_audio_mcp_server
```
HTTP 系 transport を使う場合は transport と bind を明示します。

```bash
export FLUENT_AUDIO_MCP_TRANSPORT=streamable-http
export FLUENT_AUDIO_MCP_HOST=127.0.0.1
export FLUENT_AUDIO_MCP_PORT=9110
ros2 run fa_audio_mcp fa_audio_mcp_server
```

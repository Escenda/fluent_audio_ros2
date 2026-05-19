# fa_audio_mcp

`fa_audio_mcp` は FluentAudio timeline services を MCP tools として公開する ROS2 `ament_python` package です。

公開する tool は `archive_audio_window` と `transcribe_audio` です。どちらも `<start_unix_ns>..<end_unix_ns>` 形式の numeric time range だけを受け付け、ROS2 service request に変換します。

## 目的

- Agent / MCP client から FluentAudio の音声 window archive と文字起こし service を呼び出せるようにする。
- Agent-facing な `audio_scope` を、archive 用 scope mapping と transcribe 用 scope mapping で別々に解決する。
- omitted / null / blank の `audio_scope` を tool ごとの explicit default scope key で解決する。
- ROS2 service の `success=false`、service timeout、service unavailable を tool error として返す。

## 非責務

- LLM 判断、tool 選択、会話 state machine
- 自然言語 time range 解決
- marker / turn / action id からの range 解決
- audio decode / resample / format conversion
- ASR backend 実行、audio archive 実体管理
- unsupported scope の default fallback
- transcribe scope default の暗黙有効化
- ROS service error の握りつぶし

## Tool Boundary

| Tool | ROS2 service | 主な入力 | scope mapping |
| --- | --- | --- | --- |
| `archive_audio_window` | `ArchiveAudioWindow` | `time_range`, `audio_scope`, `reason`, `related_artifact_ids`, optional format fields | `FLUENT_AUDIO_ARCHIVE_SCOPE_*` |
| `transcribe_audio` | `TranscribeAudio` | `time_range`, `audio_scope` | `FLUENT_AUDIO_TRANSCRIBE_SCOPE_*` |

`archive_audio_window` の format 省略時は adapter 側で `pcm_s16le` / `wav` / `audio/wav` を明示値として request に入れます。これは archive request の default contract であり、音声 decode や hidden conversion ではありません。

`audio_scope` が省略、null または blank の場合は、tool ごとの default scope key で解決します。archive は config loading 上の明示 default として `mic` を持ちます。transcribe は `FLUENT_AUDIO_TRANSCRIBE_DEFAULT_SCOPE` が明示されている場合だけ省略 / null / blank scope を受け付けます。default scope key が未設定、または key の指す scope mapping が未設定の場合は tool error です。

## Environment Variables

| 環境変数 | default | 内容 |
| --- | --- | --- |
| `FLUENT_AUDIO_MCP_TRANSPORT` | `stdio` | `stdio`, `sse`, `streamable-http` のいずれか |
| `FLUENT_AUDIO_MCP_HOST` | `0.0.0.0` | `sse` / `streamable-http` 用 bind host |
| `FLUENT_AUDIO_MCP_PORT` | `9110` | `sse` / `streamable-http` 用 port。正の整数 |
| `FLUENT_AUDIO_MCP_SERVICE_TIMEOUT_SEC` | `10.0` | ROS service wait / response timeout。正の数 |
| `FLUENT_AUDIO_ARCHIVE_AUDIO_WINDOW_SERVICE` | `archive_audio_window` | archive 用 ROS service 名 |
| `FLUENT_AUDIO_TRANSCRIBE_AUDIO_SERVICE` | `transcribe_audio` | transcribe 用 ROS service 名 |
| `FLUENT_AUDIO_ARCHIVE_SCOPE_MIC` | `mic` | archive tool の `mic` scope 解決先 |
| `FLUENT_AUDIO_ARCHIVE_SCOPE_SYSTEM` | 未設定 | archive tool の `system` scope 解決先 |
| `FLUENT_AUDIO_ARCHIVE_SCOPE_MIXED` | 未設定 | archive tool の `mixed` scope 解決先 |
| `FLUENT_AUDIO_ARCHIVE_DEFAULT_SCOPE` | `mic` | archive tool の null / blank `audio_scope` 解決 key |
| `FLUENT_AUDIO_TRANSCRIBE_SCOPE_MIC` | 未設定 | transcribe tool の `mic` scope 解決先 |
| `FLUENT_AUDIO_TRANSCRIBE_SCOPE_SYSTEM` | 未設定 | transcribe tool の `system` scope 解決先 |
| `FLUENT_AUDIO_TRANSCRIBE_SCOPE_MIXED` | 未設定 | transcribe tool の `mixed` scope 解決先 |
| `FLUENT_AUDIO_TRANSCRIBE_DEFAULT_SCOPE` | 未設定 | transcribe tool の null / blank `audio_scope` 解決 key |

未設定 scope は unsupported scope として fail closed します。`archive` と `transcribe` は別 mapping なので、archive の `mic` default が transcribe に流用されることはありません。

SO101 voice frontend で ASR と archive branch を分ける場合の典型値は次の通りです。

```bash
export FLUENT_AUDIO_ARCHIVE_SCOPE_MIC=mic
export FLUENT_AUDIO_TRANSCRIBE_SCOPE_MIC=audio/high_pass/mic
export FLUENT_AUDIO_TRANSCRIBE_DEFAULT_SCOPE=mic
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

この package 単体の unit tests は request validation、scope mapping、numeric time range parsing、response formatting を対象にします。実際の ROS launch / service smoke は、`fa_asr` / `fa_audio_window` service と接続した別検証です。

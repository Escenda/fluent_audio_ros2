# src/apps/agent_tools

`agent_tools/` は、FluentAudioROS2 の ROS2 topic / service 能力を外部 agent / tool runtime から呼び出すための adapter package を置く領域です。

ここに置く package は、tool schema、入力 validation、scope mapping、ROS2 service 呼び出し、service error の tool error 変換を担当します。audio DSP、codec、resample、format conversion、ASR / TTS / KWS model runtime、timeline buffer、archive storage の実体は所有しません。

## 境界

- Agent tool adapter は FluentAudio node の代わりに音声処理を実装しない。
- Agent tool adapter は LLM の判断、会話 state machine、安全停止判断を FluentAudio core に持ち込まない。
- FluentAudio node が返した `success=false` / error code を正常結果に変換しない。
- unsupported scope や未解決 time range を default 値で補完しない。
- 自然言語や marker 由来の time range 解決は、実装された resolver がある場合だけ明示的に扱う。

## Package

| Directory | Status | 内容 |
| --- | --- | --- |
| `fa_audio_mcp/` | ROS 2 ament_python package | FluentAudio timeline services を MCP tools として公開する adapter |

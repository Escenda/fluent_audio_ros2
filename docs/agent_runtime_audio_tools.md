# Agent Runtime 音声 tool 境界設計

作成日: 2026-05-20

## 0. この資料の位置づけ

この資料は、Physical AI Agent Runtime から FluentAudioROS2 の音声能力を tool として呼び出すための責務境界を定義する。

対象は次の 2 つである。

- `transcribe_audio`: 指定範囲の音声を文字起こしする tool
- `archive_audio_window`: 指定範囲の音声を証拠 clip として保存する tool

この資料は完了済み機能の一覧ではない。DSP 全分類、backend 全種、VLAbor profile 連携、Agent Runtime 実装がすべて完了したことを示す資料ではない。ここで定義するのは、次工程で実装と検証を進めるための正本境界である。

現時点の child-repo 実装として、`src/apps/agent_tools/fa_audio_mcp` に MCP adapter package が存在する。この package は `archive_audio_window` と `transcribe_audio` を MCP tools として公開し、numeric `<start_unix_ns>..<end_unix_ns>` range、archive / transcribe 別 scope mapping、null / blank / omitted `audio_scope` 用の明示 default scope key、ROS service error の tool error 変換を扱う。さらに `fluent_audio_system/config/profiles/so101_agent_audio_tools.yaml` から MCP adapter を起動できる。

実 owner node smoke として、`test/test_real_owner_graph_smoke.py::test_mcp_tools_call_real_asr_and_audio_window_owner_nodes` は Docker 内で `1 passed` として確認済みである。この smoke は `ros2 run` で実 `fa_asr_node` / `fa_audio_window_node` を起動し、in-process `FastMCP` tool から real `RosAudioTimelineClient` 経由で両 service を呼ぶ。

追加の combined launch smoke として、`src/system/fluent_audio_system/test/integration/test_combined_owner_mcp_launch_smoke.py::test_fluent_audio_system_composes_owner_and_mcp_adapter_configs` は Docker ROS2 環境で `fluent_audio_system` rebuild 後に `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/system/fluent_audio_system/test/integration/test_combined_owner_mcp_launch_smoke.py -q` を実行し、`1 passed` として確認済みである。この smoke は `config:=owner.yaml,adapter.yaml` 形式の generated owner/adapter config を `ros2 launch fluent_audio_system run.py` に渡し、fake worker / generated params を使いながら、実 `fa_asr`、`fa_audio_window` owner services と `fa_audio_mcp` adapter を同じ launch graph に載せる。起動済み MCP adapter ROS node、real `TranscribeAudio` / `ExportAudioWindow` / `ArchiveAudioWindow` service、deterministic `FLOAT32LE` ASR frame、deterministic `PCM16LE` archive frame を用意し、MCP streamable-http `/mcp` へ接続して `list_tools`、`transcribe_audio`、`archive_audio_window`、`export_audio_window` を実 call する。tool result の transcript / model ref / window ref、WAV bytes、archive metadata、export result が numeric time range に対応することと、unsupported scope が MCP error result になることを runtime behavior として検証する。

この combined launch smoke は、`fa_audio_mcp_server` executable が `fluent_audio_system` launch 経由で ROS node/process として起動し、launch-managed MCP HTTP tool-call 経路を提供することを確認する。

一方で、現在の `fluent_audio_system` `config` / `--config` はカンマ区切りの明示 config path list を受け付け、左から右の順序を保持して child-side FluentAudio config を compose できる。composition は fail closed で、空 segment、missing config、invalid YAML、`system.default_start_delay` / `system.inter_group_delay` 不一致、group id 重複、enabled node id 重複を成功扱いしない。上記 combined launch smoke は、この multi-config list を ROS runtime launch に渡した child-side `fluent_audio_system` launch-managed MCP HTTP tool-call smoke である。real `so101_voice_frontend.yaml,so101_agent_audio_tools.yaml` runtime proof、`now-10s..now` relative-time E2E、親 VLAbor profile integration、親 Agent Runtime / MCP client integration、durable storage、World Station、実 SO101 device / model provisioning は検証済みではない。

installed streamable-http transport smoke として、`src/apps/agent_tools/fa_audio_mcp/test/test_streamable_http_transport_smoke.py::test_installed_server_streamable_http_transport_calls_real_owner_nodes` は、実 `fa_asr_node` / `fa_audio_window_node`、deterministic `FLOAT32LE` ASR frame、deterministic `PCM16LE` archive frame、`ros2 run fa_audio_mcp fa_audio_mcp_server` で起動した installed server、`/mcp` の MCP streamable-http client 接続、`list_tools` / `transcribe_audio` / `archive_audio_window` の tool call、unsupported transcribe scope の MCP error result を確認済みである。

この transport smoke は child repo の installed server transport / tool behavior の検証である。自然言語 time-range resolver、親 repo の Agent Runtime / MCP client 統合、VLAbor agent 統合、World Station evidence 連携、durable storage、real SO101 runtime / device provisioning は pending である。

## 1. 基本方針

FluentAudioROS2 は音声能力を提供する。Agent Runtime はその能力を目的に応じて選び、組み合わせ、結果を読んで次の判断を行う。

FluentAudioROS2 core に Agent LLM、MCP tool 選択、保存判断、行動計画、World Station への記録判断を混ぜない。FluentAudioROS2 は音声入力、音声処理、音声解析、文字起こし、音声 window の切り出しといった能力を、ROS2 topic / service として明示的に公開する。

重要なのは、「全部を持つ」ことと「全部を一つに混ぜる」ことを分けることである。FluentAudioROS2 は format conversion、dynamics、frequency、temporal、correction / noise、spatial / channel、analysis / feature extraction、generation / transformation、routing / mixing、streaming / synchronization を扱えるべきである。ただし、それらを巨大な万能 node にまとめてはならない。それぞれの node が一つの意味を持ち、どこで音が変わり、どこで失敗したかを追える構造にする。

`transcribe_audio` と `archive_audio_window` も同じである。どちらも time range を扱うが、目的、入力表現、成功条件、失敗条件が異なる。したがって、同じ rolling window を無理に正本にしない。

## 2. 決定事項

### 2.1 `transcribe_audio` の正本 owner

`transcribe_audio` の canonical owner は `fa_asr` である。

`fa_asr` は ASR-ready な timeline を持つ。ここでいう ASR-ready とは、ASR backend に渡す前提を満たした `FLOAT32LE`、mono、設定済み sample rate、interleaved layout の音声 stream である。

`transcribe_audio` はこの ASR-ready timeline に対して numeric time range を指定し、ASR backend を実行して timestamp 付き transcript segment を返す。

`transcribe_audio` は `fa_audio_window` を秘密裏に取得元として使ってはならない。`fa_audio_window` は PCM16/WAV の証拠 window を扱う。そこから ASR-ready `FLOAT32LE` へ変換して `fa_asr` に渡すなら、PCM16/WAV から float32 への変換、sample rate 変換、channel 変換、layout 変換が hidden behavior になる。これは FluentAudioROS2 の設計方針に反する。

変換が必要なら、変換専用 node を pipeline に明示する。ASR node の中で暗黙変換してはいけない。

### 2.2 `archive_audio_window` の正本 owner

`archive_audio_window` の canonical owner は `fa_audio_window` である。

`fa_audio_window` は evidence / audio-archive 用の PCM16/WAV window を持つ。役割は、指定された time range の音声を、後から証拠として参照できる clip にすることである。

`archive_audio_window` は ASR のための意味処理をしない。文字起こしも行わない。ASR backend を呼ばない。音声 clip の保存、参照、time range の確定、範囲外や不連続 range の検出を担当する。

### 2.3 Agent Runtime / MCP adapter の責務

Agent Runtime / MCP adapter は、人間や Agent LLM が使いやすい tool input を、FluentAudioROS2 service が受け取れる deterministic な request に変換する。

Agent Runtime / MCP adapter が持つ責務は次の通りである。

| 責務 | 内容 |
| --- | --- |
| `TimeRangeSpec` 解釈 | `now-10s..now`、action marker、turn marker などを numeric media time range に解決する |
| `audio_scope` 解決 | `mic`、`system`、`mixed` などを具体的な stream identity に対応付ける |
| ROS2 service 呼び出し | `fa_asr` の `TranscribeAudio` service、`fa_audio_window` の `ArchiveAudioWindow` service を呼ぶ |
| error mapping | service の `success=false` と `error_code` を Agent Runtime の tool error として明示的に返す |
| tool contract の安定化 | Agent LLM に見せる tool schema、namespace、入力名、出力名を安定させる |

Agent Runtime / MCP adapter は ASR model をロードしない。audio decode、resample、format conversion、WAV export を自前で実装しない。FluentAudio node の代わりに音声処理を肩代わりしない。

## 3. tool contract

### 3.1 `transcribe_audio`

`transcribe_audio` は、指定された音声範囲を文字起こしする。

| 項目 | 契約 |
| --- | --- |
| Agent-facing input | `time_range`、`audio_scope`（省略時は明示 default scope key） |
| Adapter output | `time_range_spec`、`audio_scope` を持つ `TranscribeAudio` request |
| ROS2 service owner | `fa_asr` |
| 内部 timeline | ASR-ready `FLOAT32LE` timeline |
| 成功出力 | transcript segments、model ref、audio window ref、resolved time range |
| 失敗条件 | time range 未解決、window 不在、範囲外、unsupported scope、ASR 失敗、空 transcript |

`transcribe_audio` は自然言語の質問を受け取らない。Agent LLM が文字起こし結果の意味を判断したい場合は、返された transcript を読んで次の推論 turn で判断する。

現時点で確認できる実装事実は、`fa_asr` が numeric range を対象にした `TranscribeAudio` service と ASR-ready timeline を持ち、child repo の `fa_audio_mcp` が numeric range を `TranscribeAudio` request へ変換する MCP surface を持つことである。実 owner node smoke では deterministic `FLOAT32LE` frame から transcript、model ref、audio window ref が返ることまで確認済みである。自然言語 time range の解釈、親 repo の Agent Runtime 統合、World Station 連携まで完了したことは意味しない。

### 3.2 `archive_audio_window`

`archive_audio_window` は、指定された音声範囲を証拠 clip として保存する。

| 項目 | 契約 |
| --- | --- |
| Agent-facing input | `time_range`、`reason`、`related_artifact_ids`、`audio_scope`（省略時は明示 default scope key） |
| Adapter output | `time_range_spec`、`audio_scope`、`reason`、`related_artifact_ids`、codec / container / payload format を持つ `ArchiveAudioWindow` request |
| ROS2 service owner | `fa_audio_window` |
| 内部 timeline | evidence / archive 用 PCM16LE window |
| 成功出力 | audio clip ref、resolved time range |
| 失敗条件 | time range 未解決、不正 request、window 不在、範囲外、不連続 range、unsupported scope、unsupported archive format、archive 失敗 |

`archive_audio_window` は Agent LLM が保存すべきと判断した理由を受け取る。ただし、保存すべきかどうかの判断そのものは `fa_audio_window` の責務ではない。判断は Agent Runtime が行い、`fa_audio_window` は request が契約を満たすか検証して clip を作る。

現時点で確認できる実装事実は、`fa_audio_window` が PCM16LE/WAV windowing に対する export / archive service を持ち、child repo の `fa_audio_mcp` が numeric range を `ArchiveAudioWindow` request へ変換する MCP surface を持つことである。実 owner node smoke では deterministic `PCM16LE` frame から WAV clip body と `*.metadata.json` が作られ、tool result の audio clip ref / resolved time range と対応することまで確認済みである。ただし、durable object storage、World Station artifact 生成、親 repo の Agent Runtime からの tool 呼び出しまで完了したことは意味しない。

`fa_audio_window` の export / archive media identity は決定的である。`clip_id` と local path は operation、window id、window epoch、source id、stream id、resolved audio scope、resolved exported range、codec、container、payload format から導出する。archive では reason と related artifact ids も metadata identity の一部として含める。同じ request を同じ window 状態に対して繰り返すと同じ `AudioClipRef` を返し、sequence-based な別 ref は作らない。既存 deterministic target が同一 WAV bytes / metadata なら成功扱いにし、異なる WAV bytes や archive metadata conflict は `export_failed` / `archive_failed` として fail closed にする。

## 4. 2 つの timeline を分ける理由

`fa_asr` と `fa_audio_window` は、どちらも「過去の音声範囲」を扱う。しかし、同じ timeline を正本にすると責務が崩れる。

第一に、表現形式が違う。`fa_asr` の timeline は ASR backend 入力として成立する `FLOAT32LE` stream である。一方、`fa_audio_window` の timeline は証拠保存に向いた PCM16LE/WAV window である。片方から片方を作るには変換が必要になる。その変換を隠すと、どの node が音を変えたのか追えなくなる。

第二に、正しさの基準が違う。`transcribe_audio` にとって重要なのは、ASR backend が受け入れる sample rate、channel count、encoding、layout、値域、連続性である。`archive_audio_window` にとって重要なのは、証拠として後から再生、保存、参照できる clip であること、範囲が連続していること、archive format が明示されていることである。

第三に、失敗の意味が違う。ASR-ready timeline が範囲を持っていない場合、文字起こしはできない。PCM16 evidence window が範囲を持っていない場合、証拠 clip は保存できない。この 2 つは別の失敗であり、片方をもう片方で補って成功扱いにしてはいけない。

第四に、保持期間や保存粒度が将来変わり得る。ASR-ready timeline は低遅延な短期 window に寄せる可能性がある。evidence window は長めの retention、durable archive、World Station evidence ref と結び付く可能性がある。同じ lifecycle に縛ると、どちらかの設計が不自然になる。

したがって、`fa_asr` と `fa_audio_window` はそれぞれ自分の timeline を持つ。両者を対応付ける場合は、Agent Runtime / adapter が resolved time range と stream identity を明示的に渡し、必要な変換は pipeline 上の専用 node として見える形にする。

## 5. fail-closed ルール

FluentAudio core は fail closed を原則とする。

禁止するもの:

- unsupported `audio_scope` を既定 scope として扱うこと
- `now-10s..now` を node 内で曖昧に推測すること
- action marker が見つからないときに勝手に直近範囲へ置き換えること
- PCM16LE から FLOAT32LE への変換を `fa_asr` 内で隠すこと
- sample rate mismatch を ASR backend 内で吸収すること
- window 不在、範囲外、不連続 range、空 transcript を成功扱いにすること
- `success=false` を Agent Runtime 側で自然言語の警告だけにして正常結果として扱うこと

必要な data がない場合は、明示的な error code と message を返す。上位の Agent Runtime は、その error を tool failure として扱う。失敗を隠して「それらしい transcript」や「空 clip 成功」を返してはならない。

## 6. Agent Runtime / MCP adapter の設計境界

Agent Runtime / MCP adapter は、FluentAudio node の能力を Agent LLM から使える tool にする薄い境界である。ただし「薄い」とは責務が軽いという意味ではない。曖昧な自然言語入力を ROS2 service contract に落とすため、ここには明示的な validation と error mapping が必要である。

Adapter が解決する入力:

| 入力 | 解決後 |
| --- | --- |
| `now-10s..now` | numeric start / end media timestamp |
| `last_user_utterance` | 対応する turn marker の numeric range |
| `before_action:<id>` | action marker から導いた numeric range |
| `mic` | configured microphone stream identity |
| `system` | configured system-audio stream identity |
| `mixed` | configured mixed stream identity |

Adapter は範囲を推測しない。対応する marker や stream identity が見つからなければ、FluentAudio service を呼ぶ前に tool error として返す。FluentAudio service から `success=false` が返った場合も、tool error として Agent LLM に返す。

Adapter は FluentAudio node に LLM / MCP 判断ロジックを押し込まない。Agent Runtime が tool を選び、FluentAudio node が音声能力を実行する。この境界を保つことで、音声処理 node は単体で検証でき、Agent Runtime は tool orchestration と reasoning に集中できる。

## 7. テスト方針

テストは証明である。source file に特定の文字列があること、Markdown に特定の見出しがあること、launch file に特定 import があることを確認するだけのテストは、FluentAudioROS2 の信頼性を証明しない。

この境界で必要なテストは、契約とアルゴリズムの振る舞いを検証するものである。

検証すべき例:

- `TimeRangeSpec` resolver が `now-10s..now` を基準時刻から正しい numeric range に変換すること
- marker が存在しない場合に guessed range を作らず失敗すること
- `audio_scope` resolver が `mic`、`system`、`mixed` を configured stream identity にだけ解決すること
- unknown scope が default stream に落ちず、明示 error になること
- `TranscribeAudio` service が範囲外、gap、不連続、unsupported format、blank transcript を成功扱いにしないこと
- `ArchiveAudioWindow` service が unsupported codec / container / payload format を明示 error にすること
- Adapter が service error code を tool error に保存し、自然言語だけで握りつぶさないこと
- numeric range で `transcribe_audio` と `archive_audio_window` を呼び、返却された resolved time range が tool result に保持されること

テストは node の人格を守るための契約である。node ができないことをできるふりをしないこと、対応していない入力を受けたときに正しく拒否すること、成功と失敗を明確に分けることを証明する。

現在確認済みの runtime smoke は 3 つある。`src/apps/agent_tools/fa_audio_mcp/test/test_real_owner_graph_smoke.py::test_mcp_tools_call_real_asr_and_audio_window_owner_nodes` は実 `fa_asr_node` / `fa_audio_window_node`、deterministic ASR / archive frame、in-process `FastMCP`、real `RosAudioTimelineClient` を使い、成功 result と owner-node unsupported scope error を確認する。

`src/system/fluent_audio_system/test/integration/test_combined_owner_mcp_launch_smoke.py::test_fluent_audio_system_composes_owner_and_mcp_adapter_configs` は `config:=owner.yaml,adapter.yaml` 形式の generated owner/adapter config を `ros2 launch fluent_audio_system run.py` で起動し、fake worker / generated params を使いながら、`fa_asr` / `fa_audio_window` owner services と `fa_audio_mcp_server` adapter executable が同じ launch graph に載ることを確認する。さらに MCP streamable-http `/mcp` に接続し、`list_tools`、`transcribe_audio`、`archive_audio_window`、`export_audio_window` の tool call を通して、transcript / model ref / window ref / WAV bytes / archive metadata / export result と unsupported scope の MCP error result を確認する。time range は numeric range であり、`now-10s..now` の relative-time E2E ではない。

`src/system/fluent_audio_system/test/unit/test_config_schema.py` と `src/system/fluent_audio_system/test/integration/test_sample_expansion.py` では、`config` / `--config` のカンマ区切り config path list を child-side FluentAudio config として compose する package-local 検証がある。確認範囲は順序保持、required package order、空 segment、invalid YAML、timing mismatch、group id 重複、enabled node id 重複、`so101_voice_frontend.yaml,so101_agent_audio_tools.yaml` の composition である。この検証は runtime launch smoke ではない。

`src/apps/agent_tools/fa_audio_mcp/test/test_streamable_http_transport_smoke.py::test_installed_server_streamable_http_transport_calls_real_owner_nodes` は実 owner nodes と deterministic ASR / archive frame を用意し、installed `fa_audio_mcp_server` を `ros2 run fa_audio_mcp fa_audio_mcp_server` で起動する。MCP streamable-http transport で `/mcp` に接続し、`list_tools`、`transcribe_audio`、`archive_audio_window` を呼び、unsupported transcribe scope が MCP error result になることを確認する。

これらは runtime behavior の検証である。ただし、検証済みの runtime smoke は numeric time range の tool call であり、`now-10s..now` の relative-time E2E ではない。transport smoke は child repo の installed server transport / tool-call smoke であり、親 Agent Runtime / MCP client、自然言語 resolver、World Station durable storage、実 SO101 model / device provisioning はまだ検証していない。

Docker cleanup fix 後の `fa_audio_mcp` package 検証では、`colcon test --packages-select fa_audio_mcp --event-handlers console_direct+` が 79 items を収集し、`79 passed in 6.76s`、package は `7.51s` で完了している。先行して実行した real-owner smoke と streamable smoke の focused pair も `2 passed in 6.41s` として確認済みである。

`fa_audio_window` package-local 検証として、Docker/ROS 環境で `colcon build --packages-select fa_audio_window --event-handlers console_direct+ && colcon test --packages-select fa_audio_window --event-handlers console_direct+` が成功している。service contract test は 8 tests で、package test summary は 3 test binaries all passed である。この検証範囲は PCM16LE/WAV retained window の export / archive service contract、決定的 identity、既存 target conflict の fail-closed までであり、durable storage、World Station write、親 Agent Runtime MCP client、親 VLAbor profile integration、実 SO101 device / model provisioning は含まない。

## 8. 次フェーズの作業パッケージ

以下は pending work を含む作業パッケージである。child repo の MCP adapter、combined launch smoke、installed streamable-http transport smoke に実装済みの範囲は明記し、親 repo Agent Runtime、real SO101 runtime proof、storage 連携は完了扱いしない。

### 8.1 `TimeRangeSpec` resolver

child repo の runtime smoke で検証済みなのは numeric `<start_unix_ns>..<end_unix_ns>` range を service request 境界で扱う経路である。

pending として残るのは、`now-10s..now` の relative-time E2E、Agent Runtime 側の自然言語 / marker / artifact range resolver である。

対象:

- absolute timestamp range
- turn marker
- action marker
- future: World Station artifact range

失敗条件:

- syntax 不正
- marker 不在
- 基準 clock 不在
- start / end 逆転
- retention 外

### 8.2 `audio_scope` resolver

child repo の MCP adapter は、tool ごとの configured scope mapping と explicit default scope key によって `audio_scope` を service owner に渡す stream identity へ解決できる。unsupported scope、未設定 scope、解決不能な default scope は fail closed で扱う。

scope は topic 名ではない。scope は Agent-facing な論理名であり、resolver が profile / runtime config に基づいて具体的な `source_id` / `stream_id` へ変換する。

pending として残るのは、親 repo Agent Runtime / VLAbor profile からこの mapping を実運用 config として供給し、`mic`、`system`、`mixed` などの site-specific stream identity と接続することである。

### 8.3 Agent Runtime adapter / tool contract

child repo の MCP adapter package は tool input schema、ROS2 service request / response 変換、service error mapping、tool result JSON を持つ。

pending として残るのは、親 repo の Agent Runtime からこの tool を実際に呼び、Agent-facing tool contract として固定することである。

必要な契約:

- tool input schema
- resolver error schema
- service error mapping
- transcript segment result schema
- audio clip ref result schema
- model ref result schema
- resolved time range result schema

この adapter 境界は FluentAudioROS2 core に LLM 判断を持ち込まない。親 repo Agent Runtime 側の orchestration 層から呼ばれる形で検証する。

### 8.4 launch / profile integration

`fa_asr` と `fa_audio_window` は `so101_voice_frontend.yaml` から同じ FluentAudio system profile で起動できる。MCP adapter は `so101_agent_audio_tools.yaml` から別 profile として起動できる。この分割により、voice frontend の audio service owner と Agent-facing tool adapter を同じ親側 include 方式で組み合わせられる。

この作業では、ASR-ready stream と evidence stream の identity を明示する。`fa_audio_window` を `fa_asr` の暗黙入力にしない。必要な format conversion、resample、channel conversion は pipeline node として visible にする。

現時点で確認済みなのは profile expansion と required package list、child-side FluentAudio config composition、in-process `FastMCP` adapter が実 ROS graph 上の `transcribe_audio` / `archive_audio_window` service と接続できること、`config:=owner.yaml,adapter.yaml` 形式の generated owner/adapter config を `fluent_audio_system/run.py` で起動して `fa_asr` / `fa_audio_window` owner services と `fa_audio_mcp_server` adapter executable を同時に launch し、MCP streamable-http `/mcp` から `list_tools` / `transcribe_audio` / `archive_audio_window` / `export_audio_window` を呼べること、および installed `fa_audio_mcp_server` に streamable-http MCP transport で接続して `list_tools` / `transcribe_audio` / `archive_audio_window` を呼べることである。

combined launch smoke は real `so101_voice_frontend.yaml,so101_agent_audio_tools.yaml` runtime proof ではない。child-side config composition は `so101_voice_frontend.yaml` と `so101_agent_audio_tools.yaml` を package-local config / CLI として compose できることの検証であり、今回の combined launch smoke は fake worker / generated params を使う child-side generated owner/adapter config の ROS runtime launch と MCP HTTP tool-call の検証である。親側 include でどう組み合わせるか、Agent Runtime / MCP client から tool をどう呼ぶか、`now-10s..now` を E2E でどう解決するかは別検証である。

### 8.5 残る integration smoke tests

次は pending work として残る。

- 親 repo の Agent Runtime / MCP client から `transcribe_audio` / `archive_audio_window` を呼ぶ integration smoke
- World Station evidence ref と durable storage まで含む archive smoke

実 owner node smoke は、numeric range に解決済みの request で `transcribe_audio` / `archive_audio_window` が実 service owner に到達することを検証済みである。combined launch smoke は、multi-config list と `fluent_audio_system/run.py` 経由で owner services と adapter executable が同じ launch graph に載り、MCP streamable-http `/mcp` の tool call が実 owner service 経路へ到達することを検証済みである。installed streamable-http transport smoke は、child repo の installed server に `/mcp` で接続し、MCP tool call が実 owner node 経路へ到達することを検証済みである。child-side config composition は package-local config / CLI として検証済みである。ただし、`now-10s..now` relative-time E2E、Agent Runtime、durable storage、親 VLAbor profile integration は検証していない。

### 8.6 archive metadata と durable storage

`archive_audio_window` の local archive では、WAV clip と同じ basename の `*.metadata.json` を書く。この metadata は `reason`、`related_artifact_ids`、source / stream / window identity、resolved time range、audio clip ref を保持する。metadata 書き込みに失敗した場合は archive 成功扱いにしない。

local archive identity は決定的である。archive `clip_id` / path は operation、window id、window epoch、source id、stream id、resolved audio scope、resolved exported range、codec、container、payload format に加え、reason と related artifact ids から導出する。既存 WAV bytes と metadata が期待値と一致する場合は同じ `AudioClipRef` を返す。既存 WAV bytes が一致し metadata が欠損している場合は no-clobber publish で metadata を作成する。異なる WAV bytes、異なる metadata、metadata だけが存在する状態は archive failure とする。

未実装として残るのは、この local clip / metadata を durable storage や World Station evidence ref と結び付ける部分である。

検討対象:

- clip URI の永続化先
- local metadata file の durable storage への保存方法
- World Station artifact からの参照方法
- model / node / stream / codec / container / payload format metadata の上位 schema への写し方
- retention policy
- partial failure 時の rollback / cleanup

local metadata sidecar は実装済みである。durable object storage と World Station 連携は未実装である。

## 9. 非目標

この slice では次を行わない。

- concrete caller が存在しない外部推論 worker の導入
- VLM 全般の設計や実装
- FluentAudio node 内への LLM / MCP decision logic の導入
- PCM16/WAV から FLOAT32LE への hidden compatibility layer
- unsupported scope や unsupported format の自動 fallback
- 親 repo の変更
- `vlabor_ros2` の変更
- `CPP_CODING_RULES.md` / `CLAUDECODE_RULES.md` の変更

## 10. まとめ

`transcribe_audio` は `fa_asr` の ASR-ready timeline を読む。`archive_audio_window` は `fa_audio_window` の evidence / archive timeline を読む。Agent Runtime / MCP adapter は、人間や Agent LLM が指定した time range と audio scope を、FluentAudioROS2 が検証可能な numeric range と stream identity に変換する。

この分離により、FluentAudioROS2 の node は自分が何を意味しているかを保てる。文字起こし node は文字起こしに集中し、archive node は証拠保存に集中し、Agent Runtime は判断と orchestration に集中する。失敗は失敗として表に出し、対応していないものを対応済みのように見せない。それがこの境界の正しさである。

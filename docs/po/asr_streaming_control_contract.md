# ASR Streaming Control Contract PO 設計資料

## 位置づけ

この資料は、FluentAudioROS2 における ASR / KWS / TurnDetector の
streaming control contract と、NeMo RNNT streaming backend 方針を整理する
Product Owner 設計資料である。

これは実装完了報告ではない。
この資料を作成した時点で、message、service、node 実装、backend 実装、launch、profile、test、
代表検証が完了したことを意味しない。

この資料は、後続で次の作業へ落とし込むための上位契約である。

- `fa_asr` の streaming lifecycle / control state machine
- `fa_kws` の enable / suppress / reset control
- `fa_turn_detector` の turn lifecycle / commit control
- 共通 trigger schema
- `nemo_rnnt_streaming` backend contract
- `sherpa_onnx_streaming` などの別 backend と共存できる backend 境界
- 仕様書、アルゴリズム詳細説明書、テスト設計への反映

この資料に書かれた YAML は提案例であり、現時点で実装済み config であることを意味しない。

## 背景

ASR は単に audio chunk を backend に投げれば成立する処理ではない。
特に streaming ASR では、どこで stream を始め、どこまでを発話として扱い、
どの条件で transcript を final にし、どの条件で backend state を破棄するかが重要になる。

これまでの単純な表現では、次のように見えやすい。

```text
fa_in
  -> fa_sample_format
  -> fa_resample
  -> fa_frame_buffer
  -> fa_vad
  -> fa_asr
```

しかし、この表現は責務境界を誤解させる。
`fa_vad` は audio を ASR へ渡す node ではない。
`fa_vad` は発話状態や発話区間を示す side signal producer である。

`fa_asr` は、自分の backend contract に合う audio stream を読む。
VAD、KWS、TurnDetector、manual control、service request は、
ASR が stream を始めるか、継続するか、commit するか、cancel するかを決める
control side signal として扱う。

これは `fa_asr` だけの話ではない。
`fa_kws` も、いつ keyword detection を有効にするか、どの条件で抑制するかを持つ。
`fa_turn_detector` も、いつ turn detection を始め、どこで turn を確定するかを持つ。
そのため、node ごとに `vad_topic`、`kws_topic`、`turn_topic` のような専用 parameter を増やすと、
制御の意味が node ごとに分裂する。

FluentAudio では、ASR / KWS / TurnDetector が同じ考え方で control trigger を持てるようにする。
それがこの資料の中心である。

## 原則

### 全部を持つが、混ぜない

FluentAudio は音声処理に必要な領域を持つ。
format conversion、dynamics、frequency、temporal、correction / noise、spatial / channel、
analysis / feature extraction、generation / transformation、routing / mixing、
streaming / synchronization を持つ。

しかし、全部を持つことと、全部を一つの node に混ぜることは違う。

`fa_asr` は ASR node であり、VAD ではない。
`fa_kws` は keyword spotting node であり、ASR ではない。
`fa_turn_detector` は会話 turn の境界を見る node であり、audio format conversion ではない。
backend は推論 engine であり、ROS topic や control trigger を知る場所ではない。

### backend は control を知らない

backend は ROS topic を知らない。
backend は VAD topic を知らない。
backend は `control.triggers[]` を知らない。
backend は `enable_on`、`commit_on`、`cancel_on` という policy を知らない。

backend が知るべきなのは、明示された backend lifecycle だけである。

```text
start_stream
accept_audio
poll_partial
finish_stream
reset_stream
```

どの trigger で `start_stream` するか、どの timeout で `finish_stream` するか、
どの stream contract violation で `reset_stream` するかは、node または共通 control 層の責務である。

この境界を守ることで、同じ `nemo_rnnt_streaming` backend を
VAD-driven ASR、KWS-driven ASR、manual service-driven ASR のどれからでも使える。

### 未対応を隠さない

streaming ASR は stateful である。
audio frame の gap、overlap、reorder、sample rate change、epoch change を黙って飲み込むと、
backend cache は壊れる。

壊れた cache から出た transcript は、一見それらしく見えても意味が壊れている。
FluentAudio では、そのような状態を commit してはいけない。

壊れた stream は、壊れた stream として扱う。
必要なら `cancel_on` または fail closed に進む。

## Control Phase

control phase は、node が外部 signal や timeout をどの意味で受け取るかを表す。

最低限、次の phase を設計対象とする。

| phase | 意味 | 主な対象 |
| --- | --- | --- |
| `enable_on` | node の処理を開始または許可する条件 | ASR / KWS / TD |
| `disable_on` | node の処理を一時的に止める条件 | KWS / TD / ASR |
| `commit_on` | 現在の stream / turn / result を確定する条件 | ASR / TD |
| `cancel_on` | 現在の stream / turn / result を破棄する条件 | ASR / TD |
| `reset_on` | 内部状態を破棄して初期状態に戻す条件 | ASR / KWS / TD |

phase は node ごとに全てを使う必要はない。
たとえば `fa_kws` は `commit_on` を持たず、`enable_on`、`disable_on`、`reset_on` を中心に使ってよい。
`fa_turn_detector` は `commit_on` を `commit_turn_on` のような node 内部の意味へ写像してよい。

重要なのは、trigger schema を node ごとに分裂させないことである。

## Control Trigger

trigger は単一設定ではなく、複数列挙できる配列として扱う。

```text
control.<phase>.triggers[]
```

複数 trigger がある場合、どれか一つが成立すれば phase が発火する。
ただし、同時発火した場合の優先順位は明示する必要がある。
原則として、`reset_on` と `cancel_on` は `commit_on` より強い。
壊れた stream、明示 cancel、内部状態 reset が必要な状態を transcript commit で覆ってはならない。

trigger type の候補は次である。

| type | 意味 | 備考 |
| --- | --- | --- |
| `topic` | ROS topic message の field / key による trigger | VAD 専用にしない |
| `timeout` | node が保持する時刻基準による trigger | fallback ではなく明示 policy |
| `lifecycle` | node lifecycle / graph lifecycle による trigger | startup / shutdown / deactivate など |
| `stream_contract_violation` | AudioFrame stream contract 破綻による trigger | commit ではなく cancel/fail closed |
| `backend_error` | backend runtime failure による trigger | error result と fail closed を分けて設計 |
| `service_request` | service や action による明示 request | manual commit / cancel など |

### Topic Trigger

topic trigger は、特定 topic の payload を評価して発火する。
ただし、`vad_topic` のような node 専用 parameter へ固定しない。

提案例:

```yaml
type: topic
topic: voice/vad_state
stream_id: audio/vad/mic
key: is_speaking
operator: eq
value: true
```

`topic` は ROS transport の名前である。
`stream_id` は payload が属する logical stream identity である。
この二つを混同してはいけない。

`key` は message field または flattened field path を表す。
`operator` は少なくとも次を候補にする。

| operator | 意味 |
| --- | --- |
| `eq` | 等しい |
| `ne` | 等しくない |
| `gt` | より大きい |
| `gte` | 以上 |
| `lt` | より小さい |
| `lte` | 以下 |
| `exists` | field が存在する |

operator が未対応の場合は、起動時に fail closed とする。
message type、key、value の型が曖昧な場合に文字列変換で成功扱いにしてはならない。

### Timeout Trigger

`commit_on` は topic だけに依存してはならない。
VAD が end event を出さない、通信が途切れる、side signal が欠落する、といった状態で
ASR stream を永遠に開いたままにしてはいけない。

そのため、timeout trigger は default 設計に含める。

timeout は fallback ではない。
timeout は「この条件で stream を確定または破棄する」という明示 policy である。

timeout basis の候補:

| basis | 意味 | 想定 phase |
| --- | --- | --- |
| `last_audio` | 最後に有効な AudioFrame を受け取ってからの時間 | `commit_on` / `cancel_on` |
| `last_enabled_audio` | enable 状態で最後に audio を受け取ってからの時間 | `commit_on` |
| `max_stream_duration` | stream 開始からの最大継続時間 | `commit_on` / `cancel_on` |
| `no_partial_update` | partial transcript が更新されない時間 | `commit_on` / `backend_error` 寄り |
| `control_silence` | control side signal が silent になってからの時間 | `commit_on` |

timeout の扱いは phase によって意味が変わる。

- `commit_on.timeout` は現在の valid stream を final にする。
- `cancel_on.timeout` は意味が不確かな stream を破棄する。
- `reset_on.timeout` は idle 状態の古い state を掃除する。

この違いを混同してはいけない。

### Stream Contract Violation Trigger

`stream_contract_violation` は、AudioFrame stream の帳尻が合わなくなったことを表す。

対象例:

- `seq` の欠落
- `seq` の重複
- frame reorder
- `start_sample` overlap
- `start_sample` rewind
- `frame_count` と payload byte count の不一致
- `epoch` の予期しない変化
- `stream_id` の不一致
- sample rate / channel / encoding / layout の途中変更

これらは通常、commit ではない。
ASR backend cache は過去 chunk に依存するため、stream が壊れた時点で transcript の意味も壊れる可能性がある。

したがって、stream contract violation は `cancel_on` または fail closed に接続する。
後続実装では、violation の種類ごとに `cancel`、`reset`、`fatal` のどれへ進むかを明示する必要がある。

## 共通 YAML 提案例

次の YAML は提案例である。
現時点で実装済み config ではない。

```yaml
control:
  trigger_policy:
    simultaneous_priority:
      - reset_on
      - cancel_on
      - commit_on
      - disable_on
      - enable_on

  enable_on:
    default: false
    triggers:
      - type: topic
        topic: voice/vad_state
        stream_id: audio/vad/mic
        key: is_speaking
        operator: eq
        value: true

      - type: topic
        topic: voice/kws_event
        stream_id: audio/kws/mic
        key: matched
        operator: eq
        value: true

  commit_on:
    triggers:
      - type: topic
        topic: voice/vad_state
        stream_id: audio/vad/mic
        key: speech_ended
        operator: eq
        value: true

      - type: timeout
        basis: last_enabled_audio
        duration_ms: 800

      - type: timeout
        basis: max_stream_duration
        duration_ms: 30000

  cancel_on:
    triggers:
      - type: stream_contract_violation
        severity:
          - gap
          - overlap
          - reorder
          - format_change

      - type: backend_error
        severity:
          - runtime_error
          - protocol_error

      - type: service_request
        service: voice/asr_control
        command: cancel

  reset_on:
    triggers:
      - type: lifecycle
        event: deactivate

      - type: timeout
        basis: idle_state
        duration_ms: 60000
```

この構造であれば、VAD、KWS、TurnDetector、manual control を同じ control model に乗せられる。
`fa_asr` だけが `vad_topic` を持つ設計にはしない。

## ASR Lifecycle

`fa_asr` は少なくとも次の概念状態を持つ。

| state | 意味 |
| --- | --- |
| `idle` | backend stream が存在せず、audio を transcript 対象として扱っていない |
| `armed` | trigger 待ち、または短い pre-roll を保持している |
| `streaming` | backend stream が開始され、audio chunk を受け付けている |
| `committing` | final transcript を確定中 |
| `cancelling` | transcript を出さずに backend state を破棄中 |
| `failed` | node 継続が危険な失敗状態 |

state transition の概念:

```text
idle
  -- enable_on --> armed or streaming

armed
  -- valid audio --> streaming
  -- cancel_on/reset_on --> idle

streaming
  -- commit_on --> committing
  -- cancel_on --> cancelling
  -- stream_contract_violation fatal --> failed
  -- backend_error fatal --> failed

committing
  -- backend final ok --> idle
  -- backend final error --> failed or cancelling

cancelling
  -- backend reset ok --> idle

failed
  -- lifecycle reset or node restart --> idle
```

この lifecycle は設計対象であり、現時点で実装済みとは限らない。
後続実装では、各 transition がどの event を publish するかを定義する必要がある。

## Partial / Final / Error Event

streaming ASR では、途中結果と確定結果を分ける。

- `partial`: backend stream の途中結果。後続 update で変わり得る。
- `final`: commit された確定 transcript。履歴や LLM 入力に渡せる。
- `cancelled`: stream が破棄され、transcript を採用しない。
- `error`: backend error、contract violation、config error など。

`partial` を会話履歴に残すか、UI 表示だけに使うかは上位 app の責務である。
`fa_asr` は event type と stream identity、time range、reason を明示して publish する。

## KWS / TurnDetector への適用

### `fa_kws`

`fa_kws` は keyword spotting node である。
ASR のような transcript commit は持たないかもしれない。
しかし、いつ KWS を有効にするか、いつ抑制するか、いつ state を reset するかは持つ。

例:

- robot が発話中は KWS を disable する。
- ASR streaming 中は KWS を suppress する。
- safety mode 中は KWS だけ有効にする。
- device restart / epoch change で reset する。

このときも、`control.triggers[]` を使う。
`fa_kws` 専用の別 schema は作らない。

### `fa_turn_detector`

`fa_turn_detector` は turn boundary を見る。
VAD、ASR partial、ASR final、LLM response state、manual interrupt などを side signal として受ける可能性がある。

そのため、ASR と同じく trigger の複数列挙が必要になる。

例:

- VAD が speech ended を出す。
- ASR partial が一定時間更新されない。
- LLM が speaking 状態へ入る。
- user interrupt keyword が出る。
- max turn duration を超える。

これらも `topic` / `timeout` / `service_request` / `lifecycle` trigger として表現する。

## AudioFrame Stream Contract との関係

この資料は `docs/po/audio_frame_stream_contract.md` を前提にする。

streaming ASR / KWS / TurnDetector は、AudioFrame が連続していることを前提に state を持つ。
特に NeMo RNNT streaming backend は cache-aware inference を使うため、
chunk の順序と sample timeline が壊れると backend 内部状態も壊れる。

そのため、後続実装では少なくとも次の情報を検証対象にする。

- `stream_id`
- `epoch`
- `seq`
- `start_sample`
- `frame_count`
- `sample_rate`
- `channels`
- `encoding`
- `layout`

AudioFrame の `header.stamp` は時刻 anchor であり、sample continuity の代替ではない。
sample continuity は sample counter と frame count で検証する。

VAD の sample domain と ASR の sample domain が異なる場合、VAD result の sample index を
ASR sample index として直接使ってはいけない。
いったん media time range または source timeline range に解決し、
ASR stream 側の sample range として再解決する。

## NeMo RNNT Streaming Backend 方針

### なぜ model server package を前提にしないか

この設計では、特定 vendor の model server package を ASR の標準経路にしない。
動作未成立で保守負荷の大きい serving package を「正解候補」や「推奨 backend」として扱わない。

model server package が担っていた価値は、ASR model server packaging である。
しかし FluentAudio が本当に必要としている contract は、ASR backend の lifecycle と能力検証である。
したがって、外部 serving package に依存せず、NeMo model を直接扱う
`nemo_rnnt_streaming` backend を設計対象にする。

### 前提条件

`nemo_rnnt_streaming` backend は、任意の `.nemo` model を受け付ける backend ではない。
起動時に model config を読み、streaming ASR として成立するかを検証する。

最低限、次を検証する。

| 項目 | 要求 |
| --- | --- |
| model class | RNNT / Transducer 系であること |
| encoder | streaming encoder / cache-aware inference を使えること |
| decoder | RNNT decoder / joint network を持つこと |
| preprocessor | sample rate、window、feature 設定が読み取れること |
| tokenizer | vocabulary / language support が読み取れること |
| chunk | chunk size / context / right context / left context が定義できること |
| API | 呼び出し間で cache state を維持できる inference API があること |

これらを満たさない model は、起動時に fail closed とする。
offline model を chunk に切って「streaming らしく」動かして成功扱いにしてはならない。

### backend 内部状態

NeMo backend は cache-aware streaming inference の内部状態を持つ。

想定する内部状態:

- encoder cache
- convolution cache
- attention cache
- RNNT decoder state
- tokenizer / decoder state
- partial hypothesis state
- stream-local decoding context

これらは backend 内部状態である。
ROS message に出さない。
control trigger schema に出さない。
profile YAML から直接触らせない。

node は backend state の中身を知らない。
node が知るのは、backend stream が開始済みか、audio を受け付けられるか、
partial を返したか、final を返したか、reset できたかである。

### backend interface 提案例

これは概念設計であり、現時点で実装済み API ではない。

```text
start_stream(stream_contract, backend_options) -> stream_handle
accept_audio(stream_handle, audio_chunk) -> accept_result
poll_partial(stream_handle) -> partial_result | none
finish_stream(stream_handle) -> final_result
reset_stream(stream_handle, reason) -> reset_result
```

`stream_contract` には sample rate、channels、encoding、layout、stream identity、
epoch、expected chunk property などを含める。

`audio_chunk` は、node が検証済みの audio payload だけを持つ。
backend 内で sample format conversion、resample、downmix、normalize、clip を行って成功扱いにしてはならない。
必要な変換は upstream processing node の責務である。

### NeMo backend YAML 提案例

次の YAML は提案例である。
現時点で実装済み config ではない。

```yaml
backend:
  name: nemo_rnnt_streaming
  model_path: /models/asr/parakeet_streaming.nemo
  sample_rate: 16000
  channels: 1
  encoding: FLOAT32LE
  layout: interleaved

  capability:
    require_rnnt: true
    require_cache_aware_streaming: true
    fail_on_offline_model: true

  streaming:
    chunk_ms: from_model_config
    left_context: from_model_config
    right_context: from_model_config
    emit_partial: true
    max_partial_interval_ms: 300
```

`from_model_config` は、ユーザーが自由に上書きしてよい値ではない。
model config と矛盾する値が指定された場合は起動時に fail closed とする。

## sherpa-onnx との関係

`sherpa_onnx_streaming` は別 backend 候補として残せる。
sherpa-onnx は streaming ASR / VAD / KWS を同じ系統で扱える可能性があり、
local streaming backend として有力である。

ただし、この資料の主眼は backend 選定そのものではない。
主眼は、どの backend を使っても壊れない control contract を定義することである。

したがって、`nemo_rnnt_streaming` と `sherpa_onnx_streaming` は同じ node 側 lifecycle に接続する。
違いは backend capability と backend 内部 state に閉じる。

## Test Design 観点

テストは証明である。
この資料に対する後続テストは、自然言語資料の文字列を検査するものではない。
source string や Markdown string を検査しても、node が正しく振る舞うことは証明できない。

後続実装で必要なテスト観点は次である。

### State Transition

- `enable_on` で `idle` から `streaming` へ進む。
- `commit_on` で `streaming` から `committing` へ進む。
- `cancel_on` で `streaming` から `cancelling` へ進む。
- `reset_on` で state と backend stream が破棄される。
- `failed` へ進む条件が明示されている。

### Trigger Precedence

- `commit_on` と `cancel_on` が同時に成立した場合、定義された priority に従う。
- `reset_on` が成立した場合、古い partial / backend state が残らない。
- 複数 topic trigger が同時に成立しても、同じ phase を重複発火しない。

### Timeout

- `last_enabled_audio` timeout で commit する。
- `max_stream_duration` timeout で forced commit または cancel する。
- timeout は暗黙 fallback ではなく、明示 policy として event reason に残る。

### Frame Contract Violation

- gap を検出したら backend に次 chunk を渡さない。
- overlap を検出したら transcript を commit しない。
- sample rate / channel / encoding / layout の途中変更で fail closed する。
- `epoch` change を正しく stream boundary として扱う。

### Backend Capability Rejection

- RNNT / Transducer ではない model を起動時に拒否する。
- cache-aware streaming API を使えない model を起動時に拒否する。
- model config と YAML の chunk / context / sample rate が矛盾した場合に拒否する。
- backend が sample format conversion を隠して成功扱いにしない。

### Partial / Final Event

- partial は後続 update で変わり得る event として扱う。
- final は commit された transcript として扱う。
- cancel された stream は final として publish されない。
- backend error と stream contract violation の reason が区別される。

## 未決事項

この資料では、次を未決とする。

| 項目 | 未決内容 |
| --- | --- |
| 共通 control package 名 | `fa_control`、`fluent_audio_control`、既存 system package 内部実装のどれにするか |
| trigger schema の message 化 | YAML-only にするか、service/action/message でも表現するか |
| topic trigger の key path | ROS message field path の表現方法 |
| operator の型検証 | bool / int / float / string の比較規則 |
| timeout clock | ROS time / steady clock / media clock の使い分け |
| ASR pre-roll | `armed` 状態で何 ms の audio を保持するか |
| forced commit | `max_stream_duration` を commit と cancel のどちらに分類するか |
| NeMo model 対応表 | どの `.nemo` / pretrained model を正式対応にするか |
| partial event schema | `AsrEvent` / `AsrResult` の field 追加要否 |
| backend worker placement | in-process Python、subprocess worker、separate container のどれを正式経路にするか |

未決事項は、後続の Node Engineer 実装設計と代表検証で決める。
未決であることを隠して default に吸収してはならない。

## 後続作業

この資料の後続作業は次である。

1. 共通 control schema の仕様書を作る。
2. `fa_asr` 仕様書へ streaming lifecycle / trigger contract を反映する。
3. `fa_kws` / `fa_turn_detector` 仕様書へ同じ control schema を反映する。
4. `nemo_rnnt_streaming` backend docs を作る。
5. backend capability validation の実装単位を決める。
6. AudioFrame stream contract field の message 反映計画を決める。
7. state transition / timeout / frame violation / capability rejection の意味あるテストを作る。
8. representative pipeline で実音声の partial / final / cancel event を検証する。

完了判定は、この資料の存在ではなく、実装、テスト、代表検証がそろった時点で行う。

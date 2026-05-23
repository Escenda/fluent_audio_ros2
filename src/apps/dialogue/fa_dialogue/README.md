# fa_dialogue

`fa_dialogue` は FluentAudio の会話 turn lifecycle を所有する ROS 2 package です。

## Nodes

- package: `fa_dialogue`
- executable: `fa_dialogue_node`
- launch: `launch/fa_dialogue.launch.py`
- config: `config/default.yaml`
- executable: `fa_wake_ack_node`
- config: `config/wake_ack.yaml`

## Inputs

- `WakeWordResult`: wake 検出を turn 開始候補として扱う
- `VoiceActivity`: VAD の speech state を active turn 内の quiet 時間計測に使う
- `TurnEnd`: TD の発話終了判定。最終決定ではなく候補として扱う

## Outputs

- `TurnContext`: session id、user turn id、active flag を配布する
- `AsrControl`: ASR stream の `START` / `STOP` / `CANCEL` を配布する
- `TurnEndRequest`: quiet が継続した active turn について TD へ終了確認を要求する

## Boundary

`fa_dialogue` が会話 turn の唯一の所有者です。ASR backend や ASR node は wake / VAD / TD から turn を推定しません。TD は `fa_dialogue` が quiet 継続後に出す `TurnEndRequest` への判定器であり、ASR を止める最終判断は `fa_dialogue` が行います。


## Turn Timing

`fa_dialogue_node` starts ASR on wake, then waits for follow-up speech. Wake-phrase trailing silence is not treated as user-turn completion. `turn.min_active_ms` keeps the turn open for the minimum active window after wake; the SO101 profile sets this to 3000 ms. If no follow-up speech arrives, `turn.no_speech_timeout_ms` closes the turn after that window.

## Wake Ack

`fa_wake_ack_node` subscribes to `AsrControl` and publishes a short eased PCM16LE earcon when it sees `ACTION_START` with `reason=wake`. It does not own dialogue state.

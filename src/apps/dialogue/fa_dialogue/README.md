# fa_dialogue

`fa_dialogue` は FluentAudio の対話 turn context を配布する ROS 2 package です。
この最小 slice では `WakeWordResult`、`AsrResult`、`TurnEnd` を受け、現在有効な `TurnContext` を publish します。

## Node

- package: `fa_dialogue`
- executable: `fa_dialogue_node`
- launch: `launch/fa_dialogue.launch.py`
- config: `config/default.yaml`

## Inputs

- `WakeWordResult`: `detected=true` かつ keyword が空でない場合だけ turn を開始する
- `AsrResult`: active turn と session/turn が一致し、status が FINAL / TIMEOUT / ERROR の場合だけ turn を終了する
- `TurnEnd`: active turn と session/turn が一致し、`is_end=true` の場合だけ turn を終了する

## Output

- `TurnContext`: session id、user turn id、active flag を配布する

## Boundary

今回の実装は turn context publisher までです。reasoning、TTS、safety policy、external dialogue backend、robot command proposal は実装しません。
ASR、KWS、turn detector の runtime もこの package の責務外です。

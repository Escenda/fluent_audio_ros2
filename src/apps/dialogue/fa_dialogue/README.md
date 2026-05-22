# fa_dialogue

`fa_dialogue` は FluentAudio の対話 turn context を配布する ROS 2 package です。

## Node

- package: `fa_dialogue`
- executable: `fa_dialogue_node`
- launch: `launch/fa_dialogue.launch.py`
- config: `config/default.yaml`

## Inputs

- `WakeWordResult`: `detected=true` かつ keyword が空でない場合だけ turn を開始する
- `TurnEnd`: active turn と session/turn が一致し、`is_end=true` の場合だけ turn を終了する

## Output

- `TurnContext`: session id、user turn id、active flag を配布する

## Boundary

今回の実装は turn context publisher までです。reasoning、TTS、safety policy、external dialogue backend、robot command proposal は実装しません。

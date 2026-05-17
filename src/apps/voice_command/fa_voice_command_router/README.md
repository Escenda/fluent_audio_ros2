# FA Voice Command Router

`fa_voice_command_router` は音声コマンド（KWS/ASR等の結果）を受けて、起動/停止/モード切替などの状態管理を行うアプリ層ノードです。

現状はMVPとして `std_msgs/msg/String` のコマンドを購読し、状態を `std_msgs/msg/String` でPublishします（KWS/ASR自体は別パッケージで実装）。

## 起動
```bash
ros2 launch fa_voice_command_router fa_voice_command_router.launch.py
```

## コマンド例
```bash
ros2 topic pub /voice/command std_msgs/msg/String "{data: 'start'}" -1
ros2 topic pub /voice/command std_msgs/msg/String "{data: 'mode command'}" -1
ros2 topic pub /voice/command std_msgs/msg/String "{data: 'stop'}" -1
```

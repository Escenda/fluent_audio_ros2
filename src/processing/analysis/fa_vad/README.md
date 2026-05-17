# FA VAD

`fa_vad`は`fa_in`等が配信するPCMフレームを購読し、Silero VAD（PyTorch）で音声活動検知を行うノードです（オフライン前提）。

## 機能
- `fa_interfaces/msg/AudioFrame`購読（`audio/frame`）
- `audio/vad`（`std_msgs/msg/Bool`）をPublish（状態変化時）
- `voice/vad_state`（`fa_interfaces/msg/VadState`）をPublish（確率/開始/終了を含む）
- 閾値（start/end）とハングオーバーで検知を安定化

## 起動
```bash
ros2 launch fa_vad fa_vad.launch.py
```

## Runtime

PyTorch / Silero VAD は ROS package dependency ではなく、node 実行環境に明示的に provision します。`backend.model_path` は local torch.hub repository directory を指し、空または存在しない場合は起動失敗します。online download fallback はありません。`backend.execution_provider` も必須です。

## パラメータ例
```yaml
fa_vad_node:
  ros__parameters:
    target_sample_rate: 16000
    threshold_start: 0.5
    threshold_end: 0.1
    hangover_ms: 300
    backend.name: "silero"
    backend.model_path: "~/.cache/torch/hub/snakers4_silero-vad_master"
    backend.execution_provider: "cpu"
```

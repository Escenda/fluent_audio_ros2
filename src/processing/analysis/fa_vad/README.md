# FA VAD

`fa_vad`は`fa_in`等が配信するPCMフレームを購読し、外部 Silero VAD worker process で音声活動検知を行うノードです（オフライン前提）。

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

PyTorch / Silero VAD は ROS package dependency ではなく、`backend.command` で指定する外部 process 側に明示的に provision します。`backend.model_path` は local torch.hub repository directory を指し、空または存在しない場合は起動失敗します。online download fallback はありません。`backend.execution_provider` と `backend.command` も必須です。

`scripts/silero_vad_worker` は reference worker です。別 venv や別 container に同じ CLI contract の worker を置く場合は、その executable path を `backend.command` に指定します。

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
    backend.command: "/ros2_ws/install/fa_vad/lib/fa_vad/silero_vad_worker"
    backend.args:
      - "--audio"
      - "{audio}"
      - "--model"
      - "{model}"
      - "--provider"
      - "{provider}"
      - "--sample-rate"
      - "{sample_rate}"
```

# FA VAD

`fa_vad`は`fa_capture`等が配信するPCMフレームを購読し、Silero VAD（PyTorch）で音声活動検知を行うノードです（オフライン前提）。

## 機能
- `fa_interfaces/msg/AudioFrame`購読（`audio/frame`）
- `audio/vad`（`std_msgs/msg/Bool`）をPublish（状態変化時）
- `voice/vad_state`（`fa_interfaces/msg/VadState`）をPublish（確率/開始/終了を含む）
- 閾値（start/end）とハングオーバーで検知を安定化

## 起動
```bash
ros2 launch fa_vad fa_vad.launch.py
```

## パラメータ例
```yaml
fa_vad_node:
  ros__parameters:
    target_sample_rate: 16000
    threshold_start: 0.5
    threshold_end: 0.1
    hangover_ms: 300
    silero:
      # torch.hub のローカルキャッシュ（オフライン用）
      repo_dir: "~/.cache/torch/hub/snakers4_silero-vad_master"
      # オフライン前提のため既定はfalse（セットアップ時のみtrueにする運用）
      allow_online: false
```

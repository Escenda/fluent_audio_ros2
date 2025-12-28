# FV Audio VAD

`fv_audio_vad`は`fv_audio`ノードが配信するPCMフレームを購読し、簡易的なVAD（音声活動検知）をROSトピックで提供するノードです。

## 機能
- `fv_audio/msg/AudioFrame`購読（`audio/frame`）
- RMSベースの判定で`audio/vad`(std_msgs/Bool)をPublish
- ヒステリシス/ホールド時間パラメータで検知を安定化
- 将来的にWakeword/MLモデルと置き換え可能な構造

## 起動
```bash
ros2 launch fv_audio_vad fv_audio_vad_launch.py
```

## パラメータ例
```yaml
fv_audio_vad_node:
  ros__parameters:
    vad:
      threshold: 0.05
      release_ms: 200
      min_active_ms: 100
```

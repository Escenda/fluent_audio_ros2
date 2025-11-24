# FV Audio Output

`fv_audio_output`は`fv_audio/msg/AudioFrame`をALSAデバイスに出力するROS2ノードです。`fv_tts`や通知ノードがPublishする`audio/output/frame`を購読し、スピーカーへPCM16LEのまま再生します。

## 依存
- ALSA (`libasound2-dev`)
- `fv_audio`メッセージ

## 起動
```bash
ros2 run fv_audio_output fv_audio_output_node --ros-args --params-file install/fv_audio_output/share/fv_audio_output/config/default.yaml
```

主なパラメータ:
- `audio.device_id`: ALSAデバイス名（例: `default`, `hw:1,0`）
- `audio.sample_rate`, `audio.channels`, `audio.bit_depth`: フレームと一致している必要があります。
- `queue.max_frames`: バッファに保持するフレーム数。溢れると古いフレームから破棄します。

`fv_tts`と組み合わせる場合、`audio/output/frame`トピックを共有すればテキスト合成結果が自動で再生されます。

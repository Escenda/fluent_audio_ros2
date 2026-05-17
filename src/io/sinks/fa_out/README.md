# FA Out

`fa_out`は`fa_interfaces/msg/AudioFrame`をALSA raw hardware device に出力するROS2ノードです。`audio/output/frame`を購読し、スピーカーへPCM16LEのまま再生します。

## 依存
- ALSA (`libasound2-dev`)
- `fa_interfaces`メッセージ

## 起動
```bash
ros2 run fa_out fa_out_node --ros-args --params-file install/fa_out/share/fa_out/config/default.yaml
```

主なパラメータ:
- `audio.device_id`: ALSA raw hardware device id（例: `hw:1,0`）
- `audio.sample_rate`, `audio.channels`, `audio.bit_depth`: フレームと一致している必要があります。
- `queue.max_frames`: バッファに保持するフレーム数。溢れると古いフレームから破棄します。

`fa_tts`と組み合わせる場合、`audio/output/frame`トピックを共有すればテキスト合成結果が自動で再生されます。

ファイル出力やファイル再生は `fa_out` では扱いません。必要な場合は file sink/source 専用 package として切り出します。

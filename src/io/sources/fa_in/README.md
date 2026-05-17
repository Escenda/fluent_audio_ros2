# FA In

FluentAudioの音声入力ROS2ノードです。ALSAを利用して任意のマイク/ライン入力からPCMデータを取得し、ROSトピックやサービスを通じて配信します。

## 機能
- PCMデータの低遅延Publish (`audio/frame`トピック)
- デバイス列挙サービス (`list_devices`)
- ホットスワップサービス (`switch_device`)
- Diagnostic出力（XRUN/レイテンシ監視）

## ビルド
```bash
sudo apt install libasound2-dev
colcon build --packages-select fa_in
source install/setup.bash
```

## 起動
```bash
ros2 launch fa_in fa_in.launch.py
```

詳細な設計は `docs/fa_audio_design.md` を参照してください。

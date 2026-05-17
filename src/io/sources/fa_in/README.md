# FA In

FluentAudioの音声入力ROS2ノードです。ALSA の raw hardware capture source を明示的に開き、PCMデータをROSトピックやサービスを通じて配信します。

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

詳細な設計は `docs/仕様書.md`、backend 契約は `docs/backends/alsa.md` を参照してください。

# FA In

FluentAudioの音声入力ROS2ノードです。ALSA の raw hardware capture source、raw PCM file source、raw PCM UDP source を明示的に開き、PCMデータを設定されたROSトピックやサービスを通じて配信します。

## 機能
- PCMデータの低遅延Publish (`output_topic`トピック)
- デバイス列挙サービス (`list_devices`)
- ホットスワップサービス (`switch_device`)
- Diagnostic出力（XRUN/レイテンシ監視）
- `backend.name=pcm_file_reader` による raw PCM file source
- `backend.name=network_pcm_receiver` による raw PCM UDP source

## ビルド
```bash
sudo apt install libasound2-dev
colcon build --packages-select fa_in
source install/setup.bash
```

## 起動
```bash
ros2 launch fa_in fa_in.launch.py node_name:=fa_in config_file:=/path/to/fa_in.yaml
```

`config/default.yaml` は site 固有の source id を空にしています。
`audio.device_selector.identifier` または `audio.device_selector.index` を明示しない起動は fail closed します。
launch file は `node_name` / `config_file` の default を持たないため、呼び出し元 profile または system config から必ず明示します。
AudioFrame と diagnostics の QoS も `audio.qos.*` / `diagnostics.qos.*` で明示し、node 内の hidden QoS へ切り替えません。

詳細な設計は `docs/仕様書.md`、backend 契約は `docs/backends/alsa.md`、`docs/backends/pcm_file_reader.md`、`docs/backends/network_pcm_receiver.md` を参照してください。

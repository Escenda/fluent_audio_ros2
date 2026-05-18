# FA Record

`fa_record`は`audio/frame`（`fa_interfaces/msg/AudioFrame`）を購読し、`record`サービスでWAVファイルに保存するノードです。
subscription QoS は `input.qos.depth` / `input.qos.reliable` で明示し、node 内の hidden QoS へ切り替えません。

## 起動
```bash
ros2 launch fa_record fa_record.launch.py \
  node_name:=fa_record \
  config_file:=/path/to/fa_record.yaml
```

## 録音開始/停止
```bash
# 開始（保存先を指定）
ros2 service call /record fa_interfaces/srv/Record "{command: 'start', file_path: '/tmp/fa_record.wav'}"

# 停止
ros2 service call /record fa_interfaces/srv/Record "{command: 'stop', file_path: ''}"
```

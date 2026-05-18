# fa_monitor_mix

`fa_monitor_mix` は FluentAudio の routing processing package である。複数の `FLOAT32LE`
interleaved `fa_interfaces/msg/AudioFrame` stream から、device 出力とは独立した monitor bus
frame を生成する。

ROS2 node は adapter であり、sample validation / gain mix / range reject は ROS-free
`internal_monitor_mix` backend に分離している。

この package は monitor 用の合成だけを担当する。device 出力、resample、format conversion、
limiter、normalize、欠損入力への silence 合成は行わない。必要な入力が揃わない場合、stale な
frame がある場合、または mix 結果が正規化範囲を超える場合は publish せず drop として扱う。

## 入出力

| 種別 | Default | Message |
| --- | --- | --- |
| subscribe | `audio/program/frame` | `fa_interfaces/msg/AudioFrame` |
| subscribe | `audio/tts/frame` | `fa_interfaces/msg/AudioFrame` |
| publish | `audio/monitor/frame` | `fa_interfaces/msg/AudioFrame` |

## 主な parameter

| Parameter | Default | 説明 |
| --- | ---: | --- |
| `input_topics` | `["audio/program/frame", "audio/tts/frame"]` | monitor mix に必要な入力 topic |
| `input_stream_ids` | `["audio/program/frame", "audio/tts/frame"]` | 入力 frame の domain stream identity |
| `input_gains_db` | `[0.0]` | 全入力共通、または入力数と同数の gain |
| `master_index` | `0` | mix trigger と header/epoch の基準にする入力 |
| `output_topic` | `audio/monitor/frame` | publish 先 |
| `output.stream_id` | `audio/monitor/frame` | output frame の domain stream identity |
| `output.source_id` | `monitor_mix` | output frame の `source_id` |
| `max_frame_age_ms` | `100` | master 以外の latest frame 有効期限 |

## 起動

```bash
ros2 launch fa_monitor_mix fa_monitor_mix.launch.py
```

## 検証

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider src/processing/routing/fa_monitor_mix/test/unit -q
```

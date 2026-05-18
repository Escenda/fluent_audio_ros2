# fa_ducking

`fa_ducking` は FluentAudio の routing/dynamics processing package である。program stream と sidechain stream の `FLOAT32LE` interleaved `fa_interfaces/msg/AudioFrame` を購読し、recent sidechain RMS が threshold 以上の場合だけ program stream に smoothed duck gain を適用して publish する。

sidechain は制御信号としてのみ使い、output へ混ぜない。入力 frame と sample の契約違反、または変換後 sample の正規化範囲超過は frame drop として扱い、clamp や limiter で補正して publish しない。

## 入出力

| 種別 | 設定例 | Message |
| --- | --- | --- |
| subscribe | `fa_ducking/program` | `fa_interfaces/msg/AudioFrame` |
| subscribe | `fa_ducking/sidechain` | `fa_interfaces/msg/AudioFrame` |
| publish | `fa_ducking/output` | `fa_interfaces/msg/AudioFrame` |

ROS topic と `AudioFrame.stream_id` は別の identity として扱います。入力 stream は `program_stream_id` / `sidechain_stream_id`、出力 stream は `output.stream_id` で明示します。

## 主な parameter

| Parameter | 設定例 | 説明 |
| --- | ---: | --- |
| `program_stream_id` | `audio/program/frame` | program 入力 stream identity |
| `sidechain_stream_id` | `audio/sidechain/frame` | sidechain 入力 stream identity |
| `output.stream_id` | `audio/ducked/frame` | 出力 stream identity |
| `sidechain.threshold_rms` | `0.05` | ducking を有効化する sidechain RMS |
| `sidechain.max_age_ms` | `100` | sidechain frame の受信時刻ベース有効期限 |
| `ducking.gain_db` | `-12.0` | active sidechain 時の目標 gain |
| `ducking.attack_ms` | `10.0` | gain を下げる時定数 |
| `ducking.release_ms` | `250.0` | gain を `1.0` へ戻す時定数 |

## 起動

```bash
ros2 launch fa_ducking fa_ducking.launch.py config_file:=/path/to/fa_ducking.yaml
```

package launch の `config_file` は必須です。`config/default.yaml` は設定例であり、package launch から暗黙には読み込みません。

## 検証

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider src/processing/routing/fa_ducking/test/unit -q
```

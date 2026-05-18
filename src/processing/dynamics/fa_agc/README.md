# fa_agc

`fa_agc` は FluentAudio の dynamics/amplitude processing package である。正規化済み `FLOAT32LE` interleaved `fa_interfaces/msg/AudioFrame` を入力し、frame RMS に基づく automatic gain control を適用して `AudioFrame` を publish する。

この package は device gain、`fa_in`、resampling、format conversion、limiter、compressor、normalize、noise reduction を扱わない。入力 frame と sample の契約違反、または変換後 sample の正規化範囲超過は frame drop として扱い、補正して publish しない。

## 入出力例

`config/default.yaml` は単体デバッグ用の明示例であり、node の runtime default ではない。

| 種別 | Example | Message |
| --- | --- | --- |
| subscribe topic | `fa_agc/input` | `fa_interfaces/msg/AudioFrame` |
| publish topic | `fa_agc/output` | `fa_interfaces/msg/AudioFrame` |
| input stream | `audio/compressed/mic` | `AudioFrame.stream_id` |
| output stream | `audio/agc/mic` | `AudioFrame.stream_id` |

## 主な parameter 例

| Parameter | Example | 説明 |
| --- | ---: | --- |
| `agc.target_rms` | `0.1` | 目標 RMS |
| `agc.min_gain` | `0.25` | gain 下限 |
| `agc.max_gain` | `4.0` | gain 上限 |
| `agc.attack_ms` | `10.0` | gain を下げる時定数 |
| `agc.release_ms` | `250.0` | gain を上げる時定数 |

## 起動

```bash
ros2 launch fa_agc fa_agc.launch.py node_name:=fa_agc config_file:=/path/to/fa_agc.yaml
```

## 検証

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider src/processing/dynamics/fa_agc/test/unit -q
```

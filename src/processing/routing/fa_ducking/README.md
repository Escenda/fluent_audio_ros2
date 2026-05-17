# fa_ducking

`fa_ducking` は FluentAudio の routing/dynamics processing package である。program stream と sidechain stream の `FLOAT32LE` interleaved `fa_interfaces/msg/AudioFrame` を購読し、recent sidechain RMS が threshold 以上の場合だけ program stream に smoothed duck gain を適用して publish する。

sidechain は制御信号としてのみ使い、output へ混ぜない。入力 frame と sample の契約違反、または変換後 sample の正規化範囲超過は frame drop として扱い、clamp や limiter で補正して publish しない。

## 入出力

| 種別 | Default | Message |
| --- | --- | --- |
| subscribe | `audio/program/frame` | `fa_interfaces/msg/AudioFrame` |
| subscribe | `audio/sidechain/frame` | `fa_interfaces/msg/AudioFrame` |
| publish | `audio/ducked/frame` | `fa_interfaces/msg/AudioFrame` |

## 主な parameter

| Parameter | Default | 説明 |
| --- | ---: | --- |
| `sidechain.threshold_rms` | `0.05` | ducking を有効化する sidechain RMS |
| `sidechain.max_age_ms` | `100` | sidechain frame の受信時刻ベース有効期限 |
| `ducking.gain_db` | `-12.0` | active sidechain 時の目標 gain |
| `ducking.attack_ms` | `10.0` | gain を下げる時定数 |
| `ducking.release_ms` | `250.0` | gain を `1.0` へ戻す時定数 |

## 起動

```bash
ros2 launch fa_ducking fa_ducking.launch.py
```

## 検証

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider src/processing/routing/fa_ducking/test/unit -q
```

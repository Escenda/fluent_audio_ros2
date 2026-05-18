# fa_echo

`fa_echo` は FluentAudio の `AudioFrame` を受け取り、FLOAT32LE interleaved stream に内部 feedback echo を適用する ROS2 processing package です。デバイス入出力は持たず、入力 topic と出力 topic の間で sample 値のみを処理します。sample loop と delay state は ROS 非依存 backend が持ちます。

## Contract

- 入力: `fa_interfaces/msg/AudioFrame`
- 出力: `fa_interfaces/msg/AudioFrame`
- 対応 format: `FLOAT32LE` / 32 bit / interleaved
- 必須 parameter: `input_topic`, `output_topic`, `input_stream_id`, `output.stream_id`, `echo.delay_ms`, `echo.feedback_gain`, `echo.wet_gain`, `echo.dry_gain`, `expected.sample_rate`, `expected.channels`, `expected.encoding`, `expected.bit_depth`, `expected.layout`, `qos.depth`, `qos.reliable`, `diagnostics.publish_period_ms`
- source が切り替わった場合は delay state をリセットします。

## Processing

各 channel に独立した ring delay line を持ちます。

```text
output = dry_gain * input + wet_gain * delayed
next_delay_state = input + feedback_gain * delayed
```

gain は正規化しません。非 finite な設定は起動時に失敗し、契約外 frame、非 finite sample、または正規化範囲外の output/state は frame ごと drop します。

## Launch

```bash
ros2 launch fa_echo fa_echo.launch.py
```

`config/default.yaml` を標準設定として読み込みます。

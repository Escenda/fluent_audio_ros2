# no_runtime_backend

## 1. 目的

`fa_trim` は backend を持たない temporal processing node である。
処理は ROS2 node 内で byte range copy と contract validation のみを行う。

## 2. 依存 runtime

- C++17
- `rclcpp`
- `diagnostic_msgs`
- `fa_interfaces`

外部 process、Python venv、container worker、model artifact は不要である。

## 3. Input / Output Format

input / output とも `fa_interfaces/msg/AudioFrame` を使用する。
format は `FLOAT32LE`、32bit、interleaved に限定する。

## 4. Failure Policy

backend fallback は存在しない。parameter または frame contract が不正な場合は、起動失敗または frame drop とする。

## 5. Diagnostics

runtime backend 固有 diagnostics はない。node diagnostics の counter を正とする。

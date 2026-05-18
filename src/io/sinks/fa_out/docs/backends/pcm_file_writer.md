# PCM File Writer Backend

## Backend Name

`pcm_file_writer`

## Contract

`backend.name=pcm_file_writer` の backend です。`fa_out` が受け付けた
`AudioFrame` payload を、明示された file target へ raw PCM bytes として
write します。

backend は ROS2 adapter ではないため、`rclcpp`、`fa_interfaces`、ROS message
header を include しません。ROS topic、stream identity、playback lifecycle は
`fa_out` node が扱います。

## Required Config

- `file.path`
- `overwrite.enabled`
- `audio.encoding`
- `audio.channels`
- `audio.bit_depth`

`PCM16LE/16`、`PCM32LE/32`、`FLOAT32LE/32` だけを受け付けます。

## Forbidden

- hidden codec encode
- hidden resample
- hidden channel conversion
- hidden gain / normalize / limiter
- output path guessing
- parent directory creation

## Failure Conditions

- empty `file.path`
- missing parent directory
- target exists while `overwrite.enabled=false`
- directory target
- unsupported encoding / bit depth pair
- write error

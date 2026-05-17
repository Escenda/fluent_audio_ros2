# local_command Backend

## Backend Name

`local_command`

## Runtime

外部 command を subprocess として実行します。ROS2 node は engine の Python package を import しません。

## Input

- mono float samples
- sample rate

## Command Contract

backend は一時 WAV file path を command に渡し、stdout または output file から transcript を読む構造です。

## Failure Conditions

- command path missing
- non-zero exit
- timeout
- empty transcript when empty result is disallowed

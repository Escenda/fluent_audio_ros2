# pyopenjtalk backend

## 目的

`pyopenjtalk` は local Japanese TTS backend である。

## 入力

- text
- voice id

## 出力

- sample rate
- mono PCM16LE bytes

## runtime dependency

`pyopenjtalk` は ROS package dependency ではなく、node 実行環境に明示的に provision する。`backend.name=pyopenjtalk` が選択された状態で未導入なら起動失敗する。

Backend は ROS2 topic、ROS message、`rclpy` を知らない。`AudioFrame` への変換は node が行う。

## 失敗条件

- empty text
- invalid voice
- synthesis failure
- runtime import failure

失敗時に cloud TTS や別 voice へ自動切替しない。

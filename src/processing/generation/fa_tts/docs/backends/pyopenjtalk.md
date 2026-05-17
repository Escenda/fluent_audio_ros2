# pyopenjtalk backend

## 目的

`pyopenjtalk` は local Japanese TTS backend である。

## 入力

- text
- voice id
- volume dB

## 出力

- sample rate
- mono PCM16LE bytes
- RMS / peak

## runtime dependency

`pyopenjtalk` は ROS package dependency ではなく、node 実行環境に明示的に provision する。未導入なら import 時点で起動失敗する。

## 失敗条件

- empty text
- invalid voice
- synthesis failure
- runtime import failure

失敗時に cloud TTS や別 voice へ自動切替しない。

# network_streamer backend

## 目的

`network_streamer` は validated audio chunk を外部 endpoint へ送る backend である。ROS2 topic を知らず、endpoint / codec / transport config と chunk のみを扱う。

## 入力

- endpoint URI
- transport
- codec or PCM format
- audio chunk

## 失敗条件

- endpoint URI が空
- unsupported transport
- unsupported encoding
- connect / write failure
- packetize failure

失敗時に local file や別 endpoint へ送る fallback は禁止する。

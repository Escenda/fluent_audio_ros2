# network_streamer backend

## 目的

`network_streamer` は validated audio chunk を外部 endpoint へ送る backend である。ROS2 topic を知らず、endpoint / codec / transport config と chunk のみを扱う。

`network_streamer` は network sink backend であり、`src/streaming` 配下の transport stabilization node ではない。jitter buffer、clock drift correction、PLC、time alignment はこの backend に混ぜない。

## 入力

- endpoint URI
- transport
- codec or PCM format
- `PCM16LE` / 16-bit / positive sample_rate / positive channels / interleaved audio chunk

current raw PCM packet contract は accepted chunk 1 件を明示 transport の packet 1 件として送ることです。codec mode を使う場合も codec は config で明示し、hidden resample、downmix、channel conversion、bit-depth conversion、format conversion は行わない。

unsupported endpoint / transport / encoding / bit_depth / sample_rate / channels / layout / packet shape は startup fail、runtime fatal、または explicit error result にする。jitter buffer、clock drift correction、PLC、time alignment は `src/streaming` 側の明示 node で扱い、この backend に隠さない。

## 失敗条件

- endpoint URI が空
- unsupported transport
- unsupported encoding
- connect / write failure
- packetize failure

失敗時に local file や別 endpoint へ送る fallback は禁止する。

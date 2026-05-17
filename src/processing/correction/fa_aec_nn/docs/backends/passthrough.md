# passthrough backend

## 目的

`passthrough` は `fa_aec_nn` の wiring 検証用 backend である。実際の echo suppression は行わない。

## 入力

- validated `AudioFrame`

## 出力

- 入力と同じ `AudioFrame`

## 失敗条件

- node 側の format validation に失敗した frame

## 注意

この backend は NN AEC の代替ではない。未実装 model backend を `passthrough` に自動変換してはならない。

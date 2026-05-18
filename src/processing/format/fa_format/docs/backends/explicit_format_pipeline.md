# explicit_format_pipeline backend

`explicit_format_pipeline` は format conversion stage list を検証し、launch/composition 用の展開結果へ変換する backend contract である。

## Required Config

- ordered `stages`
- stage package
- stage launch file
- stage params file
- stage input topic
- stage output topic
- expected input format
- expected output format

## Contract

backend は stage list を validation し、展開可能な pipeline description を返す。
audio samples、ROS2 topic payload、device handle は扱わない。

## Forbidden

- missing stage の skip
- adjacent format mismatch の自動補正
- implicit resample
- implicit channel conversion
- implicit encode/decode
- default params file の推測


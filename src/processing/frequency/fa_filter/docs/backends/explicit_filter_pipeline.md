# explicit_filter_pipeline backend

`explicit_filter_pipeline` は frequency stage list を検証し、launch/composition 用の展開結果へ変換する backend contract である。

## Required Config

- ordered `stages`
- stage package
- stage launch file
- stage params file
- stage input topic
- stage output topic
- stage input stream id
- stage output stream id
- expected input format
- expected output format
- frequency-specific parameters

## Forbidden

- missing stage の skip
- adjacent format mismatch の自動補正
- adjacent stream identity mismatch の自動補正
- ROS topic と `AudioFrame.stream_id` の兼用
- hidden resample
- hidden gain / normalize
- default params file の推測

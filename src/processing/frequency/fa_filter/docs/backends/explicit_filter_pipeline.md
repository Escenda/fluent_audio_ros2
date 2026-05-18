# explicit_filter_pipeline backend

`explicit_filter_pipeline` は frequency stage list を検証し、launch/composition 用の展開結果へ変換する backend contract である。

## Required Config

- ordered `stages`
- stage package
- stage launch file
- stage params file
- stage input topic
- stage output topic
- expected input format
- expected output format
- frequency-specific parameters

## Forbidden

- missing stage の skip
- adjacent format mismatch の自動補正
- hidden resample
- hidden gain / normalize
- default params file の推測


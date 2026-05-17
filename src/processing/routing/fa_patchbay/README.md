# fa_patchbay

`fa_patchbay` は FluentAudio の `processing/routing` に属する静的 patchbay node です。
設定された `input_topics[i] -> output_topics[i]` の route に従い、`AudioFrame` を複製して publish します。

## 責務

- 明示された input topic から `AudioFrame` を subscribe する
- 同じ input topic を共有する複数 route へ frame を複製する
- frame の `stream_id` だけを output topic に更新する
- 無効な config は起動時に fail closed する
- 無効な runtime frame は warning と diagnostics counter を残して drop する

## 非責務

- DSP
- format conversion
- gain / normalize
- mixing
- fallback route 推定
- device I/O

## 起動例

```bash
ros2 launch fa_patchbay fa_patchbay.launch.py
```

既定 config は `config/default.yaml` です。詳細な契約は `docs/仕様書.md` を参照してください。

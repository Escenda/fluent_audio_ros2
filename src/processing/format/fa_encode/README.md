# fa_encode

`fa_encode` は PCM `AudioFrame` を明示的な encoded audio contract へ変換する format processing node の設計ディレクトリです。

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_codec_encoder.md`

この node は `fa_out` に codec encode を隠さないための専用段です。speaker sink、network sink、file sink、resample、gain、channel conversion は扱いません。

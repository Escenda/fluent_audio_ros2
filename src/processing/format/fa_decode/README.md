# fa_decode

`fa_decode` は encoded audio を明示的な PCM `AudioFrame` 契約へ変換する format processing node です。

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/external_codec_decoder.md`

この node は `fa_in` に file/network decode を隠さないための専用段です。device/source adapter、resample、gain、channel conversion は扱いません。

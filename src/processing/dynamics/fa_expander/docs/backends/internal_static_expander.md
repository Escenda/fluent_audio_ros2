# internal_static_expander

## Backend Scope

`internal_static_expander` は外部 DSP ライブラリを使わない、フレーム内サンプル単位の静的下向きエキスパンダである。

この backend は以下を行わない。

- デバイス入出力
- リサンプリング
- sample format 変換
- channel 変換
- compressor / limiter / gate / normalize / filter / denoise

## Failure Policy

設定不正は起動時に例外で fail closed する。実行時フレームの契約違反、非 finite サンプル、正規化範囲外サンプル、出力不正はフレーム単位で drop し、warning を出す。

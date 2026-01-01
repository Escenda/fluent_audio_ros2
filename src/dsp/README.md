# src/dsp

ノイズ抑制・AEC・リサンプリング・ミキサ等のDSP系パッケージを配置します。

実装済み（骨組み）:
- `fa_resample`: 48k→16k などのリサンプル（AEC用のmic/refも想定）
- `fa_aec_linear`: 線形AEC（現状は単純減算の仮実装）
- `fa_aec_nn`: 残差抑圧（現状はパススルー）
- `fa_ns`: ノイズ抑制（DTLN/ONNX。パススルーも可）
- `fa_mix`: PCM16LE のミキサ（MVP）

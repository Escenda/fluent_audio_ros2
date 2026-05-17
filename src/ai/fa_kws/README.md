# fa_kws

`fa_kws` は sherpa-onnx のローカル KWS モデルを使うウェイクワード/キーワードスポッティングノードです。

## 入出力

- Sub: `audio/frame` (`fa_interfaces/msg/AudioFrame`)
- Sub: `voice/vad_state` (`fa_interfaces/msg/VadState`)
- Pub: `voice/wake_word` (`fa_interfaces/msg/WakeWordResult`)

## モデル

`config/default.yaml` の `model.encoder` / `model.decoder` / `model.joiner` / `model.tokens` / `kws.keywords_file` は必須です。空または存在しないパスを指定した場合、ノードは起動時に失敗します。`backend.execution_provider` も必須で、空または未対応 provider は sherpa-onnx C API に渡す前に失敗します。KWS は VAD state を必須入力として扱い、未受信または `vad.max_age_ms` を超えた stale state では audio frame を処理しません。

ビルド時には sherpa-onnx C API が必要です。標準パスに無い場合は `SHERPA_ONNX_PREFIX` 環境変数または CMake cache で install prefix を指定してください。

Contract test だけを実行する環境では、`-DFA_KWS_SHERPA_ONNX=OFF` を明示できます。sherpa-onnx C API を必須にする runtime build では `-DFA_KWS_SHERPA_ONNX=ON` を使います。`AUTO` または `OFF` で sherpa-onnx が組み込まれない場合でも、別 backend や dummy backend へ暗黙に切り替えることはありません。

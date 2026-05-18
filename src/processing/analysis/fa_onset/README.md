# fa_onset

`fa_onset` は `FLOAT32LE` mono `AudioFrame` から spectral flux による onset strength と onset flag を生成する非 AI analysis node です。

この package は VAD、KWS、ASR、Turn Detector の推論を実行しません。model runtime は持たず、ROS-free な `internal_spectral_flux` backend で deterministic onset measurement だけを行います。

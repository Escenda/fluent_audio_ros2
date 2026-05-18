# fa_pitch

`fa_pitch` は `FLOAT32LE` mono `AudioFrame` から autocorrelation による fundamental frequency と voiced flag を生成する非 AI analysis node です。

この package は VAD、KWS、ASR、Turn Detector の推論を実行しません。model runtime は持たず、ROS-free な `internal_autocorrelation` backend で deterministic pitch measurement だけを行います。

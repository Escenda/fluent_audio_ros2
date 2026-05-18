# fa_cqt

`fa_cqt` は `FLOAT32LE` mono `AudioFrame` から complex constant-Q feature matrix を生成する非 AI analysis node です。

この package は VAD、KWS、ASR、Turn Detector の推論を実行しません。model runtime は持たず、ROS-free な `internal_cqt` backend で deterministic feature extraction だけを行います。

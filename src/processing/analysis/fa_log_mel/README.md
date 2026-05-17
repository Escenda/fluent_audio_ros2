# fa_log_mel

`fa_log_mel` は `FLOAT32LE` mono `AudioFrame` から log-mel feature matrix を生成する非 AI analysis node です。

この package は VAD、KWS、ASR、Turn Detector の推論を実行しません。model runtime は持たず、ROS-free な `internal_log_mel` backend で deterministic feature extraction だけを行います。

# fa_mfcc

`fa_mfcc` は `FLOAT32LE` mono `AudioFrame` から MFCC feature matrix を生成する非 AI analysis node です。

この package は VAD、KWS、ASR、Turn Detector の推論を実行しません。model runtime は持たず、ROS-free な `internal_mfcc` backend で deterministic feature extraction だけを行います。

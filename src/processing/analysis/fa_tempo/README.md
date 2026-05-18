# fa_tempo

`fa_tempo` は `FLOAT32LE` mono `AudioFrame` から onset envelope と autocorrelation による tempo を測定する非 AI analysis node です。

この package は VAD、KWS、ASR、Turn Detector の推論を実行しません。model runtime は持たず、ROS-free な `internal_onset_autocorrelation` backend で deterministic tempo measurement だけを行います。

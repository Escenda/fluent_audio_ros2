# waveform_declip backend

`waveform_declip` は clipped region detection と waveform reconstruction を行う backend contract である。

## Required Config

- `backend.name`
- `clipping.threshold`
- `reconstruction.window_frames`
- `reconstruction.confidence_threshold`

## Forbidden

- hidden limiter / compressor
- hidden normalize
- clipping failure の pass-through
- ROS2 topic/message dependency inside backend


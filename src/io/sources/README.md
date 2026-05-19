# src/io/sources

Input source adapters live here. They open explicit sources and publish audio
frames without resampling, gain, filtering, denoise, or model inference.
`fa_in` owns ALSA capture, raw PCM file, and raw PCM UDP source backends. It
does not hide decode, jitter buffering, packet loss concealment, clock drift
correction, gain, resampling, or model inference inside those source adapters.

`fa_file_in` and `fa_network_in` are design-map directories for future
standalone source adapters. They are intentionally not ROS 2 packages yet.
Current system configs must use `fa_in` with explicit `backend.name` values for
raw PCM file and raw PCM UDP sources.

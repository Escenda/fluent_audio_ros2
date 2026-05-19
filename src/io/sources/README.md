# src/io/sources

Input source adapters live here. They open explicit sources and publish audio
frames without resampling, gain, filtering, denoise, or model inference.
`fa_in` owns ALSA capture, raw PCM file, and raw PCM UDP source backends. It
does not hide decode, jitter buffering, packet loss concealment, clock drift
correction, gain, resampling, or model inference inside those source adapters.

# FluentAudio Processing

`src/processing` contains DSP and feature-extraction nodes that transform,
analyze, generate, or route audio streams. Device input and output stay in
`src/io`. AI model nodes stay in `src/ai`. Real-time transport stabilization
stays in `src/streaming`. Application policy and dialogue orchestration stay in
`src/apps`.

The direct children of this directory are the stable processing taxonomy. New
ROS 2 packages must be placed under exactly one category:

| Directory | Responsibility |
| --- | --- |
| `format/` | Representation changes such as sample rate, channels, bit depth, interleave, and codec boundaries. |
| `dynamics/` | Amplitude and loudness control such as gain, normalize, compressor, limiter, gate, and AGC. |
| `frequency/` | Frequency-domain shaping such as EQ, low-pass, high-pass, notch, de-esser, and spectral filters. |
| `temporal/` | Time-axis editing such as trim, silence removal, delay, reverb, fade, and windowing. |
| `correction/` | Input repair such as denoise, AEC, dereverberation, declip, hum removal, and DC offset removal. |
| `spatial/` | Spatial and channel processing such as pan, downmix, upmix, beamforming, and source separation. |
| `analysis/` | Non-AI feature extraction and measurements such as STFT, Mel spectrogram, MFCC, and loudness. |
| `generation/` | Audio generation and conversion such as TTS, voice conversion, neural codecs, and vocoders. |
| `routing/` | Signal routing such as mixer, bus routing, ducking, loopback, monitor mix, and patchbay. |

Each ROS 2 package under these categories should keep the standard package
contract:

- `README.md` as a short entry point.
- `docs/仕様書.md` for external behavior.
- `docs/アルゴリズム詳細説明書.md` for internal processing.
- `docs/テスト設計.md` for spec-to-test mapping.
- `docs/backends/` for backend-specific contracts when the package has engines.
- `test/unit`, `test/integration`, `test/launch`, and `test/fixtures`.

Processing nodes do not infer device configuration, rewrite system configs, or
hide missing model/backend requirements with fallback behavior. Missing required
parameters fail closed at the package boundary.

## Package Status

The taxonomy directories are stable even when a category has no concrete ROS 2
package yet. A directory without `package.xml` is a roadmap/category placeholder
and must not be counted as implemented package coverage.

| Category | Current ROS 2 packages | Status |
| --- | --- | --- |
| `format/` | `fa_resample`, `fa_sample_format`, `fa_bit_depth`, `fa_channel_convert`, `fa_interleave` | implemented packages |
| `dynamics/` | `fa_gain`, `fa_normalize`, `fa_compressor`, `fa_limiter`, `fa_expander`, `fa_noise_gate`, `fa_agc` | implemented packages |
| `frequency/` | `fa_eq`, `fa_high_pass`, `fa_low_pass`, `fa_band_pass`, `fa_notch`, `fa_deesser` | implemented packages |
| `temporal/` | `fa_delay`, `fa_echo`, `fa_reverb`, `fa_trim`, `fa_silence_removal`, `fa_fade`, `fa_window` | implemented packages |
| `correction/` | `fa_aec_linear`, `fa_aec_nn`, `fa_denoise`, `fa_declick`, `fa_hum`, `fa_dc_offset_removal` | implemented packages |
| `spatial/` | `fa_pan`, `fa_stereo_widening`, `fa_downmix`, `fa_upmix`, `fa_beamforming` | implemented packages |
| `analysis/` | `fa_log_mel`, `fa_loudness`, `fa_stft` | implemented packages |
| `generation/` | `fa_tts` | implemented package |
| `routing/` | `fa_mix`, `fa_bus_router`, `fa_sidechain`, `fa_ducking`, `fa_monitor_mix`, `fa_loopback`, `fa_patchbay` | implemented packages |

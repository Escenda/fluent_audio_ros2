# FluentAudio Processing

`src/processing` contains audio graph nodes that transform, analyze, generate,
route, or stabilize audio streams. Device input and output stay in `src/io`.
Application policy and dialogue orchestration stay in `src/apps`.

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
| `analysis/` | Audio analysis and model input such as VAD, KWS, ASR, turn detection, spectrograms, and embeddings. |
| `generation/` | Audio generation and conversion such as TTS, voice conversion, neural codecs, and vocoders. |
| `routing/` | Signal routing such as mixer, bus routing, ducking, loopback, monitor mix, and patchbay. |
| `streaming/` | Real-time transport stability such as buffers, jitter buffers, drift correction, latency compensation, and overlap-add. |

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

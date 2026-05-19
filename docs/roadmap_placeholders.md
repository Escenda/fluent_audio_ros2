# Roadmap Placeholder Classification

This document classifies README-only directories that intentionally do not have
`package.xml`. They are not ROS 2 packages and must not be referenced from
`fluent_audio_system` configs until promoted.

Promotion requires a package-local contract: `package.xml`, build metadata,
`config`, `launch`, `docs/仕様書.md`, `docs/アルゴリズム詳細説明書.md`,
`docs/テスト設計.md`, `docs/backends/`, and package-local unit, integration,
launch, and fixture tests.

## Promotion Policy

| Class | Meaning | Action |
| --- | --- | --- |
| near-term package candidate | Needed for microphone, recognition, or robot audio reliability and narrow enough to package independently. | Design and implement as a real ROS 2 package before it is used in a system profile. |
| explicit roadmap-only | Useful conceptually, but either covered by an existing package/backend today or too broad for the current voice interaction path. | Keep README-only and non-buildable. Do not add `package.xml`, `config`, or `launch`. |
| umbrella/category placeholder | Naming bucket, not a single runtime node. | Keep non-buildable unless a concrete node contract replaces the umbrella concept. |

## Near-Term Package Candidates

| Directory | Reason |
| --- | --- |
| `src/ai/fa_speaker/` | Speaker embedding / identification is a recognition primitive likely to feed dialogue context and user-specific behavior. |
| `src/ai/fa_sed/` | Sound event detection is an audio recognition primitive distinct from VAD/KWS/ASR/TD. |
| `src/processing/correction/fa_dereverb/` | Room echo correction is directly relevant to robot microphones and ASR stability. |
| `src/processing/correction/fa_declip/` | Clipped microphone input should fail or be repaired explicitly before model input. |
| `src/processing/correction/fa_debreath/` | Breath/noise suppression may be useful for near-field microphones and dialogue recordings. |
| `src/processing/correction/fa_wind/` | Wind handling is relevant for mobile or fan-adjacent robot deployments. |
| `src/processing/frequency/fa_spectral_subtraction/` | Classical noise reduction backend useful before VAD/ASR and narrow enough for a standalone package. |
| `src/processing/frequency/fa_wiener/` | Statistical noise filtering backend useful before VAD/ASR and narrow enough for a standalone package. |
| `src/processing/spatial/fa_source_separation/` | Useful for multi-speaker or robot-environment separation, but should be promoted only with an explicit backend contract. |

## Explicit Roadmap-Only

| Directory | Reason to keep non-buildable for now |
| --- | --- |
| `src/io/sources/fa_file_in/` | File input is currently an explicit `fa_in` backend. Split only if file semantics exceed source-adapter scope. |
| `src/io/sources/fa_network_in/` | Network input is currently an explicit `fa_in` backend. Split only if transport lifecycle becomes package-owned. |
| `src/io/sinks/fa_file_out/` | File output is currently an explicit `fa_out` backend. Split only if file sink lifecycle becomes package-owned. |
| `src/io/sinks/fa_network_out/` | Network output is currently an explicit `fa_out` backend. Split only if packetization/transport becomes package-owned. |
| `src/apps/safety/fa_safety_policy/` | Safety policy must be specified against robot-control contracts before package promotion. |
| `src/processing/generation/fa_voice_conversion/` | Not required for the current voice interaction core. |
| `src/processing/generation/fa_speech_enhancement/` | Overlaps correction/denoise until a concrete generation-style enhancement contract is defined. |
| `src/processing/generation/fa_speech_separation/` | Should wait for source-separation and backend-runtime contracts. |
| `src/processing/generation/fa_speech_translation/` | Not required for the current Japanese voice interaction core. |
| `src/processing/generation/fa_music_source_separation/` | Outside the near-term robot voice path. |
| `src/processing/generation/fa_neural_codec/` | Requires a token/latent representation contract before ROS package promotion. |
| `src/processing/generation/fa_neural_vocoder/` | Requires a spectrogram/latent-to-waveform contract before ROS package promotion. |
| `src/processing/generation/fa_super_resolution/` | Not required before baseline input/output and recognition paths are stable. |
| `src/processing/spatial/fa_binaural/` | Output rendering feature; not needed before baseline speaker output works. |
| `src/processing/spatial/fa_ambisonics/` | 360-degree audio field processing is outside the current voice interaction core. |
| `src/processing/temporal/fa_time_stretch/` | Creative/time-editing feature; not needed before real-time voice recognition path. |
| `src/processing/temporal/fa_pitch_shift/` | Creative/time-editing feature; not needed before real-time voice recognition path. |

## Umbrella / Category Placeholders

| Directory | Reason |
| --- | --- |
| `src/processing/format/fa_format/` | Format conversion is already split into concrete packages such as sample format, bit depth, channels, interleave, encode, decode, and resample. |
| `src/processing/frequency/fa_filter/` | Filtering is already split into concrete packages such as high-pass, low-pass, band-pass, notch, EQ, and de-esser. |

## System Config Rule

`fluent_audio_system` may include only directories with `package.xml`. A roadmap
placeholder listed here is invalid in system config even when disabled, because
disabled profiles should not imply launchability.

# Audio AI

`src/ai` contains audio AI nodes whose main responsibility is model inference or
model-backed decision output. These packages may depend on local executables,
native inference engines, external workers, or explicit cloud/API backends, but
their runtime engines stay behind package-local `backends/` boundaries.

AI nodes do not perform source/sink device binding, hidden resampling, hidden
channel conversion, or transport buffering. Required format conversion belongs
in `src/processing/format`; DSP cleanup belongs in `src/processing`; real-time
buffering and synchronization belong in `src/streaming`.

## Package Status

Only directories with `package.xml` are ROS 2 packages.

| Directory | Status |
| --- | --- |
| `fa_kws/` | ROS 2 package |
| `fa_turn_detector/` | ROS 2 package |
| `fa_audio_embedding/` | ROS 2 package |
| `fa_sed/` | roadmap placeholder; not a ROS 2 package |
| `fa_speaker/` | roadmap placeholder; not a ROS 2 package |

# FluentAudio ROS2 C++ Coding Rules

This document defines C++ rules for `fluent_audio_ros2`. Parent-repository
rules still apply. If a local rule conflicts with the parent `AGENTS.md`, follow
the stricter fail-closed rule.

## 1. Responsibility Boundaries

- `fa_in` is a source adapter. It opens an explicit source backend and publishes
  `AudioFrame` bytes plus metadata. It must not resample, decode, downmix, apply
  gain, normalize, denoise, run VAD/KWS/ASR/TD, or hide source selection.
- `fa_out` is a sink adapter. It subscribes to explicit `AudioFrame` contracts
  and writes to an explicit sink backend. It must not encode/decode, resample,
  normalize, limit, mix, route, or generate speech.
- DSP and format work belongs in `src/processing/<category>/`.
- Buffering, jitter, packet loss concealment, clock drift correction, time
  alignment, chunk overlap, and overlap-add nodes belong in `src/streaming/`.
- VAD, KWS, ASR, turn detection, SED, speaker, and audio embedding model nodes
  belong in `src/ai/`.
- Backend code must not include `rclcpp`, `fa_interfaces`, or ROS message types.

## 2. Fail Closed

Do not add fallback that changes meaning. Missing device, missing backend,
unknown backend, missing model, unsupported format, failed health check, invalid
topic/stream identity, stale required source, or backend runtime failure must be
visible as one of:

- startup failure
- runtime fatal shutdown
- explicit frame rejection with a concrete reason
- explicit error result where the message contract requires downstream notice

Do not silently continue with default devices, guessed model paths, empty
results, zero-filled audio, latest available data, implicit CPU fallback, or
automatic format conversion.

## 3. Package And Backend Layout

C++ ROS packages should use:

```text
fa_<name>/
  include/fa_<name>/
  src/
  src/backends/ or include/fa_<name>/backends/
  config/default.yaml
  launch/
  docs/
  test/unit
  test/integration
  test/launch
  test/fixtures
```

Use a ROS-free backend interface for device, file, network, DSP engine, and
external worker adapters. The ROS node owns parameters, publishers,
subscriptions, services, lifecycle, and message conversion at the boundary.

## 4. Parameters And Contracts

- Declare required ROS parameters without runtime defaults that would hide
  missing config.
- Validate types explicitly before converting signed ROS parameter values to
  unsigned or size types.
- Keep ROS topic names and stream identities separate. A `*_topic` value must not
  be treated as an `AudioFrame.stream_id`.
- Validate `source_id`, `stream_id`, sample rate, channel count, encoding, bit
  depth, layout, QoS depth, and timing values before processing.
- Reject unsupported or non-byte-aligned payloads before touching backend state.
- Do not normalize, clip, clamp, resample, downmix, or bit-depth convert inside a
  node that is not the explicit processing node for that operation.

## 5. C++ Style

- Use C++17.
- Put public interfaces under `include/<package>/`.
- Prefer one focused class or tightly related group per header/source pair.
- Use `#pragma once` in headers.
- Include order in `.cpp`: own header, standard library, third-party/system,
  ROS2/message headers, local package headers.
- Use package namespaces such as `fa_in`, `fa_out`, `fa_kws`, and nested
  `backends` where useful.
- Class names use `CamelCase`; functions and variables use `camelCase` or
  `snake_case` consistently with the local file; private data members end in
  `_`.
- Use `constexpr` for constants.
- Keep functions focused. Split long logic when it obscures validation,
  conversion, or lifecycle boundaries.

## 6. Error Handling

- Throw `std::runtime_error` or `std::invalid_argument` for invalid config and
  backend contract violations.
- In runtime callbacks, convert unrecoverable backend failures to fatal log plus
  ROS shutdown. Do not keep consuming audio after a required backend fails.
- Frame rejection is allowed only for per-frame invalid input such as stream
  mismatch, stale optional gate state, or malformed frame payload. Log the
  reason.
- Do not add warning-only continuation for required resources.

## 7. Tests

Every behavior-changing C++ change must include or update at least one focused
test:

- ROS-free backend unit/gtest for backend validation and algorithms.
- Python static contract tests only for repository layout or source-boundary
  invariants that cannot be exercised cheaply at runtime.
- Launch tests for required launch arguments and absence of package-local
  fallback config.
- Integration tests for external worker protocols and graph-level contracts.

Run the narrow package test first. For ROS package changes, verify through
`colcon build` and `colcon test` for the touched package and its direct message
dependencies.

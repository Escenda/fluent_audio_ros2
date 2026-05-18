# baseline_linear

`baseline_linear` is the package-local C++ backend used by `fa_aec_linear`.
It does not know ROS2 topics, ROS2 messages, launch files, or parameter files.
It is not a full production AEC engine.

## Input Contract

- mic byte buffer
- reference byte buffer
- configured channels, encoding, bit depth, and cancel gain
- same sample count after decode
- `expected_channels > 0`; wildcard channel validation is not allowed
- supported format pairs are `PCM16LE/16` and `FLOAT32LE/32`

## Output Contract

When mic and reference buffers are valid and aligned, the backend subtracts
`cancel_gain * reference` from mic samples and returns an encoded buffer with
the configured explicit encoding / bit depth pair. It does not emit `PCM32LE/32`
and does not clamp out-of-range samples.

## Failure Policy

The backend returns explicit reject status instead of rewriting output when:

- mic/reference buffer is empty
- mic/reference byte length is not aligned
- configured format is unsupported
- FLOAT32LE mic/reference sample is non-finite or outside `[-1.0, 1.0]`
- mic/reference sample counts differ
- output samples are non-finite or outside `[-1.0, 1.0]`

Future production AEC engines should be added as explicit processing backends
or separate packages rather than hidden fallback paths in this baseline node.

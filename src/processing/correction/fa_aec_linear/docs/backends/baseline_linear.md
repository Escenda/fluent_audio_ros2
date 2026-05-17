# baseline_linear

`fa_aec_linear` currently uses an internal baseline linear subtraction path.
It is not an external backend and does not attempt to be a full production AEC
engine.

## Input Contract

- mic `AudioFrame`
- reference `AudioFrame`
- same sample rate, channels, sample count, encoding, and bit depth contract
- supported format pairs are `PCM16LE/16` and `FLOAT32LE/32`
- explicit `ref_timeout_ms`
- `reference_failure_policy: "drop"`

## Output Contract

When mic and reference frames are valid and aligned, the node subtracts
`cancel_gain * reference` from mic samples and publishes a new `AudioFrame`
with the original explicit encoding / bit depth pair. It does not emit
`PCM32LE/32` and does not clamp out-of-range samples.

## Failure Policy

The node drops frames instead of passing mic through when:

- reference is missing
- reference is stale
- reference format differs from mic format
- mic/reference decode fails
- no aligned samples are available
- mic/reference sample counts differ
- output samples are non-finite or outside `[-1.0, 1.0]`
- node `enabled=false`

Future production AEC engines should be added as explicit processing backends
or separate packages rather than hidden fallback paths in this baseline node.

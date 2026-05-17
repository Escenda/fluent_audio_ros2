# baseline_linear

`fa_aec_linear` currently uses an internal baseline linear subtraction path.
It is not an external backend and does not attempt to be a full production AEC
engine.

## Input Contract

- mic `AudioFrame`
- reference `AudioFrame`
- same sample rate, channels, bit depth, and encoding contract
- explicit `ref_timeout_ms`
- `reference_failure_policy: "drop"`

## Output Contract

When mic and reference frames are valid and aligned, the node subtracts
`cancel_gain * reference` from mic samples and publishes a new `AudioFrame`.

## Failure Policy

The node drops frames instead of passing mic through when:

- reference is missing
- reference is stale
- reference format differs from mic format
- mic/reference decode fails
- no aligned samples are available
- node `enabled=false`

Future production AEC engines should be added as explicit processing backends
or separate packages rather than hidden fallback paths in this baseline node.

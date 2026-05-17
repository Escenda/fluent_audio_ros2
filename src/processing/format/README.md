# Format Processing

This category contains nodes that change audio representation without intending
to change the semantic content of the sound.

Examples:

- sample-rate conversion
- bit-depth conversion
- channel conversion
- interleaved/planar conversion
- PCM value-range conversion
- encode/decode boundaries

`fa_in` and `fa_out` do not perform these conversions internally. If a pipeline
requires a specific representation, add an explicit format node in the system
config.

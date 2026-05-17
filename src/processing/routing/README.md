# Routing Processing

This category contains nodes that manage signal paths rather than changing one
waveform in isolation.

Examples:

- mixer
- bus routing
- sidechain
- ducking
- monitor mix
- loopback
- patchbay

Routing nodes make graph structure explicit. They must not rewrite system
configs or infer hidden source/sink wiring at runtime.

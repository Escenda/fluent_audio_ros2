# no_runtime_backend

`fluent_audio_system` is a launch orchestration package and has no runtime
audio/model backend.

It reads a system config, validates enabled groups/nodes, expands explicit
`${share:...}` paths, and creates ROS 2 launch actions.

## Boundary

- It does not open audio devices.
- It does not inspect hardware.
- It does not select model or DSP backends.
- It does not synthesize or rewrite params files.
- It does not fall back from missing params files to inline-only parameters.

## Failure Policy

Missing `config`, missing enabled-node `params_file`, unresolved package share
paths, invalid schema, and invalid remappings fail at launch expansion time.

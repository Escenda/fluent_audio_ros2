# no_runtime_backend

`fa_interfaces` is an interface-only package and has no runtime backend.

It defines ROS 2 messages and services shared by FluentAudio packages. Backend
selection belongs to the package that owns a node, model runtime, device
adapter, or external worker.

## Boundary

- No node is launched from this package.
- No audio processing or inference runs here.
- No device, model, or cloud API dependency is declared here.

## Failure Policy

Invalid runtime payloads are validated by the receiving node package. This
package does not add defaults or fallback values to message fields.

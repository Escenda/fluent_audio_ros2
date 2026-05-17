# fa_network_out

Roadmap directory for the network sink adapter that sends incoming `AudioFrame`
payloads to an explicitly configured endpoint.

This is not a ROS 2 package yet. Do not add `package.xml` until the sink adapter
specification, backend documentation, launch contract, and tests are in place.
Transport stabilization belongs in `src/streaming`, not inside this adapter.

# fa_file_out

Roadmap directory for the file sink adapter that writes incoming `AudioFrame`
payloads to an explicitly configured file target.

This is not a ROS 2 package yet. Do not add `package.xml` until the sink adapter
specification, backend documentation, launch contract, and tests are in place.
Encoding must remain an explicit `fa_encode` pipeline stage when the output file
format differs from the incoming frame contract.

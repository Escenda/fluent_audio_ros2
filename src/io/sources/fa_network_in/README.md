# fa_network_in

Roadmap directory for the network source adapter that receives audio from an
explicit network endpoint and publishes its contract as `AudioFrame`.

This is not a ROS 2 package yet. Do not add `package.xml` until the source
adapter specification, backend documentation, launch contract, and tests are in
place. Jitter buffering, clock drift correction, and packet loss concealment
belong in `src/streaming`, not inside this adapter.

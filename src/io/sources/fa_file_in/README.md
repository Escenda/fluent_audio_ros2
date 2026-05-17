# fa_file_in

Roadmap directory for the file source adapter that reads explicitly configured
audio files and publishes their contract as `AudioFrame`.

This is not a ROS 2 package yet. Do not add `package.xml` until the source
adapter specification, backend documentation, launch contract, and tests are in
place. File decode must remain an explicit `fa_decode` pipeline stage when the
input file is not already in the configured frame contract.

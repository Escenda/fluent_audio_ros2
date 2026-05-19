# src/io

Audio I/O packages are split by responsibility:

- `sources/`: input source adapters such as `fa_in`.
- `sinks/`: output sink adapters such as `fa_out`.
- `utilities/`: I/O utilities such as recording and network streaming.

Format conversion, gain, filters, denoise, AEC, routing, and buffering are
processing nodes, not hidden behavior inside sources or sinks.

## Package Status

Only directories with `package.xml` are ROS 2 packages.

| Directory | Status |
| --- | --- |
| `sources/fa_in/` | ROS 2 package |
| `sinks/fa_out/` | ROS 2 package |
| `utilities/fa_record/` | ROS 2 package |
| `utilities/fa_stream/` | ROS 2 package |

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
| `sources/fa_file_in/` | roadmap directory; current file source implementation lives in `fa_in` as the `pcm_file_reader` backend |
| `sources/fa_network_in/` | roadmap directory; current network source implementation lives in `fa_in` as the `network_pcm_receiver` backend |
| `sinks/fa_out/` | ROS 2 package |
| `sinks/fa_file_out/` | roadmap directory; current file sink implementation lives in `fa_out` as the `pcm_file_writer` backend |
| `sinks/fa_network_out/` | roadmap directory; current network sink implementation lives in `fa_out` as the `network_pcm_sender` backend |
| `utilities/fa_record/` | ROS 2 package |
| `utilities/fa_stream/` | ROS 2 package |

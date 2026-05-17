# src/io

Audio I/O packages are split by responsibility:

- `sources/`: input source adapters such as `fa_in`.
- `sinks/`: output sink adapters such as `fa_out`.
- `utilities/`: I/O utilities such as recording and network streaming.

Format conversion, gain, filters, denoise, AEC, routing, and buffering are
processing nodes, not hidden behavior inside sources or sinks.

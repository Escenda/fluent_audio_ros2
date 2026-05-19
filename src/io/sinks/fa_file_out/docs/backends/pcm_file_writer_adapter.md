# pcm_file_writer_adapter

`pcm_file_writer_adapter` is the future backend contract for a standalone
`fa_file_out` package.

The current implementation lives in `fa_out` as the `pcm_file_writer` sink
backend. This document exists to keep the design-map directory explicit without
declaring a second ROS 2 package.

Backend responsibilities:

- open an explicit raw PCM file output path
- validate target byte contract
- write received chunks exactly as configured

Backend non-responsibilities:

- codec encode
- sample-rate conversion
- channel conversion
- limiter / normalize / fade
- ROS 2 topic or message handling

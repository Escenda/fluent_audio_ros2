# pcm_file_reader_adapter

`pcm_file_reader_adapter` is the future backend contract for a standalone
`fa_file_in` package.

The current implementation lives in `fa_in` as the `pcm_file_reader` source
backend. This document exists to keep the design-map directory explicit without
declaring a second ROS 2 package.

Backend responsibilities:

- open an explicit raw PCM file path
- validate byte contract before publishing starts
- read fixed-size chunks
- report EOF explicitly

Backend non-responsibilities:

- codec decode
- sample-rate conversion
- channel conversion
- gain / normalize / denoise
- ROS 2 topic or message handling

# pcm_file_writer backend

`pcm_file_writer` は incoming `AudioFrame` payload を明示 file target へ raw PCM として write する backend contract である。

## Required Config

- `backend.name`
- `file.path`
- `expected.sample_rate`
- `expected.channels`
- `expected.encoding`
- `expected.bit_depth`
- `expected.layout`
- `overwrite.enabled`

## Forbidden

- hidden codec encode
- hidden resample
- hidden channel conversion
- hidden gain / normalize / limiter
- output path guessing

## Runtime Boundary

The backend is a ROS-free file adapter. It opens an explicitly configured target
file, writes raw bytes supplied by the node, flushes after each write, and
reports the number of bytes accepted by the backend.

It does not create parent directories and does not write WAV/MP3/AAC/FLAC
containers. Encoded output belongs in `fa_encode` before this sink receives the
frame.

## Failure Policy

- empty `file.path`: fail closed
- missing parent directory: fail closed
- target exists while `overwrite.enabled=false`: fail closed
- directory target: fail closed
- write error: fail closed

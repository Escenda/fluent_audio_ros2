# pcm_file_reader backend

`pcm_file_reader` は raw PCM file を source adapter として読む backend contract である。

## Required Config

- `backend.name`
- `file.path`
- `expected.sample_rate`
- `expected.channels`
- `expected.encoding`
- `expected.bit_depth`
- `expected.layout`

## Forbidden

- hidden decode
- hidden resample
- hidden gain / normalize
- missing file fallback

## Runtime Boundary

The backend is a ROS-free file adapter. It opens a regular file in binary mode,
reports its byte size, reads requested byte chunks, and can seek back to the
start when the node explicitly enables `playback.loop`.

It does not parse WAV/MP3/AAC/FLAC containers. Encoded input belongs in
`fa_decode` before the pipeline reaches this source contract.

## Failure Policy

- empty `file.path`: fail closed
- missing or non-regular file: fail closed
- empty file: fail closed
- read error: fail closed
- file byte size not divisible by expected bytes-per-frame: fail closed

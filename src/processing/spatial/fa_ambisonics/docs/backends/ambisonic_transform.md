# ambisonic_transform backend

`ambisonic_transform` は ambisonic order / normalization / channel ordering を明示して channel transform を行う backend contract である。

## Required Config

- `backend.name`
- `ambisonics.order`
- `ambisonics.normalization`
- `ambisonics.channel_order`
- `coordinate.convention`

## Forbidden

- channel ordering guessing
- hidden binaural rendering
- hidden beamforming
- ROS2 topic/message dependency inside backend


#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace fa_audio_window
{

struct AudioFormat
{
  std::string encoding{};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  uint32_t bit_depth{0};
  std::string layout{};
};

inline bool sameAudioFormat(const AudioFormat & left, const AudioFormat & right)
{
  return left.encoding == right.encoding &&
    left.sample_rate == right.sample_rate &&
    left.channels == right.channels &&
    left.bit_depth == right.bit_depth &&
    left.layout == right.layout;
}

inline size_t bytesPerSample(const AudioFormat & format)
{
  if (format.bit_depth == 0 || format.bit_depth % 8u != 0u) {
    throw std::runtime_error("bit_depth must be a positive multiple of 8");
  }
  return static_cast<size_t>(format.bit_depth / 8u);
}

inline size_t bytesPerSampleFrame(const AudioFormat & format)
{
  if (format.channels == 0u) {
    throw std::runtime_error("channels must be > 0");
  }
  return bytesPerSample(format) * static_cast<size_t>(format.channels);
}

}  // namespace fa_audio_window

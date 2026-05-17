#include "fa_kws/audio_utils.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace fa_kws
{

std::vector<float> frameToCanonicalFloat(const fa_interfaces::msg::AudioFrame &msg)
{
  if (msg.data.empty()) {
    return {};
  }

  if (msg.channels != 1u) {
    throw std::invalid_argument(
      "AudioFrame channels must be 1, got " + std::to_string(msg.channels));
  }
  if (msg.bit_depth != 32u) {
    throw std::invalid_argument(
      "AudioFrame bit_depth must be 32, got " + std::to_string(msg.bit_depth));
  }
  if (msg.data.size() % sizeof(float) != 0u) {
    throw std::invalid_argument("AudioFrame float32 data length is not byte-aligned");
  }

  const std::size_t count = msg.data.size() / sizeof(float);
  std::vector<float> samples(count);
  for (std::size_t i = 0; i < count; ++i) {
    float value = 0.0f;
    std::memcpy(&value, msg.data.data() + i * sizeof(float), sizeof(float));
    if (!std::isfinite(value)) {
      throw std::invalid_argument("AudioFrame contains non-finite samples");
    }
    if (value < -1.0f || value > 1.0f) {
      throw std::invalid_argument("AudioFrame samples must be normalized to [-1.0, 1.0]");
    }
    samples[i] = value;
  }
  return samples;
}

}  // namespace fa_kws

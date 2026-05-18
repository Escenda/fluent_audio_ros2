#include "fa_kws/audio_utils.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace fa_kws
{

namespace
{
constexpr const char * kFloat32Encoding = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
}

std::vector<float> frameToCanonicalFloat(
  const fa_interfaces::msg::AudioFrame &msg,
  const std::string &expected_source_id,
  const std::string &expected_stream_id)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    throw std::invalid_argument("AudioFrame source_id and stream_id are required");
  }
  if (expected_source_id.empty()) {
    throw std::invalid_argument("expected_source_id is required");
  }
  if (expected_stream_id.empty()) {
    throw std::invalid_argument("expected_stream_id is required");
  }
  if (msg.source_id != expected_source_id) {
    throw std::invalid_argument("AudioFrame source_id must match expected_source_id");
  }
  if (msg.stream_id != expected_stream_id) {
    throw std::invalid_argument("AudioFrame stream_id must match expected_stream_id");
  }
  if (msg.channels != 1u) {
    throw std::invalid_argument(
      "AudioFrame channels must be 1, got " + std::to_string(msg.channels));
  }
  if (msg.layout != kInterleavedLayout) {
    throw std::invalid_argument("AudioFrame layout must be interleaved, got " + msg.layout);
  }
  if (msg.encoding != kFloat32Encoding) {
    throw std::invalid_argument("AudioFrame encoding must be FLOAT32LE, got " + msg.encoding);
  }
  if (msg.bit_depth != 32u) {
    throw std::invalid_argument(
      "AudioFrame bit_depth must be 32, got " + std::to_string(msg.bit_depth));
  }
  if (msg.data.empty()) {
    throw std::invalid_argument("AudioFrame data is required");
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

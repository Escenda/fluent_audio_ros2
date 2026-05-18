#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_aec_nn::backends
{

struct AudioChunk
{
  int sample_rate = -1;
  int channels = -1;
  int bit_depth = -1;
  std::string encoding;
  std::string layout;
  const uint8_t * data = nullptr;
  size_t data_size = 0;
};

struct ProcessedAudioChunk
{
  int sample_rate = -1;
  int channels = -1;
  int bit_depth = -1;
  std::string encoding;
  std::string layout;
  std::vector<uint8_t> data;
};

inline bool isAlignedPcmPayload(
  const size_t data_size,
  const int channels,
  const int bit_depth)
{
  if (data_size == 0 || channels <= 0 || bit_depth <= 0 || (bit_depth % 8) != 0) {
    return false;
  }
  const auto bytes_per_sample = static_cast<size_t>(bit_depth / 8);
  const auto bytes_per_frame = static_cast<size_t>(channels) * bytes_per_sample;
  return bytes_per_frame > 0 && (data_size % bytes_per_frame) == 0;
}

inline std::string validateProcessedAudioChunk(
  const AudioChunk & input,
  const ProcessedAudioChunk & output)
{
  if (output.sample_rate != input.sample_rate) {
    return "backend output sample_rate must match input sample_rate";
  }
  if (output.channels != input.channels) {
    return "backend output channels must match input channels";
  }
  if (output.encoding != input.encoding) {
    return "backend output encoding must match input encoding";
  }
  if (output.bit_depth != input.bit_depth) {
    return "backend output bit_depth must match input bit_depth";
  }
  if (output.layout != input.layout) {
    return "backend output layout must match input layout";
  }
  if (!isAlignedPcmPayload(output.data.size(), output.channels, output.bit_depth)) {
    return "backend output audio data must be non-empty and PCM frame aligned";
  }
  if (output.data.size() != input.data_size) {
    return "backend output audio data size must match input audio data size";
  }
  return "";
}

class AecNnBackend
{
public:
  virtual ~AecNnBackend() = default;

  virtual const char * name() const = 0;
  virtual ProcessedAudioChunk process(const AudioChunk & chunk) = 0;
};

}  // namespace fa_aec_nn::backends

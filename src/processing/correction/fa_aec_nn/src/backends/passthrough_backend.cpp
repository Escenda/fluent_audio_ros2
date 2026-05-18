#include "fa_aec_nn/backends/passthrough_backend.hpp"

#include <stdexcept>
#include <vector>

namespace fa_aec_nn::backends
{

const char * PassthroughBackend::name() const
{
  return "passthrough";
}

ProcessedAudioChunk PassthroughBackend::process(const AudioChunk & chunk)
{
  if (chunk.data == nullptr || chunk.data_size == 0) {
    throw std::invalid_argument("passthrough backend requires non-empty audio data");
  }
  ProcessedAudioChunk output;
  output.sample_rate = chunk.sample_rate;
  output.channels = chunk.channels;
  output.bit_depth = chunk.bit_depth;
  output.encoding = chunk.encoding;
  output.layout = chunk.layout;
  output.data = std::vector<uint8_t>(chunk.data, chunk.data + chunk.data_size);
  return output;
}

}  // namespace fa_aec_nn::backends

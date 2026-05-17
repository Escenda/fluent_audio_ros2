#include "fa_aec_nn/backends/passthrough_backend.hpp"

#include <stdexcept>
#include <vector>

namespace fa_aec_nn::backends
{

const char * PassthroughBackend::name() const
{
  return "passthrough";
}

std::vector<uint8_t> PassthroughBackend::process(const AudioChunk & chunk)
{
  if (chunk.data == nullptr || chunk.data_size == 0) {
    throw std::invalid_argument("passthrough backend requires non-empty audio data");
  }
  return std::vector<uint8_t>(chunk.data, chunk.data + chunk.data_size);
}

}  // namespace fa_aec_nn::backends

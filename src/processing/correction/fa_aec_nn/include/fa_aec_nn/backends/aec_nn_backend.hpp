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
  std::string layout;
  const uint8_t * data = nullptr;
  size_t data_size = 0;
};

class AecNnBackend
{
public:
  virtual ~AecNnBackend() = default;

  virtual const char * name() const = 0;
  virtual std::vector<uint8_t> process(const AudioChunk & chunk) = 0;
};

}  // namespace fa_aec_nn::backends

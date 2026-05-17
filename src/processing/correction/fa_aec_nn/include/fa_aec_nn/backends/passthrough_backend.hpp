#pragma once

#include "fa_aec_nn/backends/aec_nn_backend.hpp"

namespace fa_aec_nn::backends
{

class PassthroughBackend final : public AecNnBackend
{
public:
  const char * name() const override;
  std::vector<uint8_t> process(const AudioChunk & chunk) override;
};

}  // namespace fa_aec_nn::backends

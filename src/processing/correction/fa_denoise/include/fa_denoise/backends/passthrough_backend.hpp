#pragma once

#include "fa_denoise/backends/denoise_backend.hpp"

namespace fa_denoise::backends
{

class PassthroughBackend final : public DenoiseBackend
{
public:
  static constexpr const char * kName = "passthrough";

  PassthroughBackend(AudioFormat expected_format, AudioFormat output_format);

  [[nodiscard]] const char * name() const override;
  [[nodiscard]] ProcessResult process(const AudioBuffer & input) override;

private:
  AudioFormat expected_format_;
  AudioFormat output_format_;
};

}  // namespace fa_denoise::backends

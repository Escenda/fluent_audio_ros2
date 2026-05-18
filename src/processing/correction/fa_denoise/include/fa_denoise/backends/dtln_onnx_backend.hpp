#pragma once

#include <memory>

#include "fa_denoise/backends/denoise_backend.hpp"
#include "fa_denoise/backends/dtln_onnx_engine.hpp"

namespace fa_denoise::backends
{

struct DtlnOnnxBackendConfig
{
  AudioFormat expected_format;
  AudioFormat output_format;
  DtlnOnnxConfig engine_config;
};

class DtlnOnnxBackend final : public DenoiseBackend
{
public:
  static constexpr const char * kName = "dtln_onnx";

  explicit DtlnOnnxBackend(const DtlnOnnxBackendConfig & config);
  ~DtlnOnnxBackend() override;

  DtlnOnnxBackend(const DtlnOnnxBackend &) = delete;
  DtlnOnnxBackend & operator=(const DtlnOnnxBackend &) = delete;
  DtlnOnnxBackend(DtlnOnnxBackend &&) = delete;
  DtlnOnnxBackend & operator=(DtlnOnnxBackend &&) = delete;

  [[nodiscard]] const char * name() const override;
  [[nodiscard]] ProcessResult process(const AudioBuffer & input) override;
  [[nodiscard]] size_t pendingInputSamples() const override;

private:
  AudioFormat expected_format_;
  AudioFormat output_format_;
  DtlnOnnxConfig engine_config_;
  std::unique_ptr<DtlnOnnxEngine> engine_;
};

}  // namespace fa_denoise::backends

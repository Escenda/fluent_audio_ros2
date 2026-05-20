#pragma once

#include <memory>
#include <string>

#include "fa_resample/backends/resampler_backend.hpp"

namespace fa_resample::backends
{

enum class BackendKind
{
  kInternalLinearResampler,
  kSpeexDsp,
  kSoxr,
};

enum class SoxrQuality
{
  kQq,
  kLq,
  kMq,
  kHq,
  kVhq,
};

struct BackendSelection
{
  BackendKind kind{BackendKind::kInternalLinearResampler};
  std::string name;
  int target_sample_rate{-1};
  int speex_quality{-1};
  SoxrQuality soxr_quality{SoxrQuality::kMq};
  std::string quality_label;
};

BackendKind parseBackendKind(const std::string & name);
int validateSpeexDspQuality(int quality);
SoxrQuality parseSoxrQuality(const std::string & quality);
std::string soxrQualityName(SoxrQuality quality);
unsigned long soxrQualityRecipe(SoxrQuality quality);
std::unique_ptr<ResamplerBackend> createResamplerBackend(const BackendSelection & selection);

}  // namespace fa_resample::backends

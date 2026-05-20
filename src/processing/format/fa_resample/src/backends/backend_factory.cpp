#include "fa_resample/backends/backend_factory.hpp"

#include "fa_resample/backends/internal_linear_resampler.hpp"
#include "fa_resample/backends/soxr_resampler.hpp"
#include "fa_resample/backends/speexdsp_resampler.hpp"

#include <stdexcept>
#include <string>

namespace fa_resample::backends
{

BackendKind parseBackendKind(const std::string & name)
{
  if (name == InternalLinearResamplerBackend::kName) {
    return BackendKind::kInternalLinearResampler;
  }
  if (name == SpeexDspResamplerBackend::kName) {
    return BackendKind::kSpeexDsp;
  }
  if (name == SoxrResamplerBackend::kName) {
    return BackendKind::kSoxr;
  }
  throw std::runtime_error(
    "backend.name must be one of internal_linear_resampler, speexdsp, soxr");
}

int validateSpeexDspQuality(const int quality)
{
  if (quality < 0 || quality > 10) {
    throw std::runtime_error("backend.quality for speexdsp must be an integer in 0..10");
  }
  return quality;
}

SoxrQuality parseSoxrQuality(const std::string & quality)
{
  if (quality == "QQ") {
    return SoxrQuality::kQq;
  }
  if (quality == "LQ") {
    return SoxrQuality::kLq;
  }
  if (quality == "MQ") {
    return SoxrQuality::kMq;
  }
  if (quality == "HQ") {
    return SoxrQuality::kHq;
  }
  if (quality == "VHQ") {
    return SoxrQuality::kVhq;
  }
  throw std::runtime_error("backend.quality for soxr must be one of QQ, LQ, MQ, HQ, VHQ");
}

std::string soxrQualityName(const SoxrQuality quality)
{
  switch (quality) {
    case SoxrQuality::kQq:
      return "QQ";
    case SoxrQuality::kLq:
      return "LQ";
    case SoxrQuality::kMq:
      return "MQ";
    case SoxrQuality::kHq:
      return "HQ";
    case SoxrQuality::kVhq:
      return "VHQ";
  }
  throw std::logic_error("unhandled soxr quality");
}

unsigned long soxrQualityRecipe(const SoxrQuality quality)
{
  switch (quality) {
    case SoxrQuality::kQq:
      return 0UL;
    case SoxrQuality::kLq:
      return 1UL;
    case SoxrQuality::kMq:
      return 2UL;
    case SoxrQuality::kHq:
      return 4UL;
    case SoxrQuality::kVhq:
      return 6UL;
  }
  throw std::logic_error("unhandled soxr quality");
}

std::unique_ptr<ResamplerBackend> createResamplerBackend(const BackendSelection & selection)
{
  if (selection.target_sample_rate <= 0) {
    throw std::runtime_error("target_sample_rate must be > 0");
  }

  switch (selection.kind) {
    case BackendKind::kInternalLinearResampler:
      return std::make_unique<InternalLinearResamplerBackend>(
        InternalLinearResamplerConfig{selection.target_sample_rate});
    case BackendKind::kSpeexDsp:
      return std::make_unique<SpeexDspResamplerBackend>(
        SpeexDspResamplerConfig{
          selection.target_sample_rate,
          validateSpeexDspQuality(selection.speex_quality),
          "libspeexdsp.so.1"});
    case BackendKind::kSoxr:
      return std::make_unique<SoxrResamplerBackend>(
        SoxrResamplerConfig{selection.target_sample_rate, selection.soxr_quality, "libsoxr.so.0"});
  }
  throw std::logic_error("unhandled resampler backend kind");
}

}  // namespace fa_resample::backends

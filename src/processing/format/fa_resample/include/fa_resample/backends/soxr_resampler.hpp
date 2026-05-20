#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "fa_resample/backends/backend_factory.hpp"
#include "fa_resample/backends/resampler_backend.hpp"

namespace fa_resample::backends
{

struct SoxrResamplerConfig
{
  int target_sample_rate{-1};
  SoxrQuality quality{SoxrQuality::kMq};
  std::string library_path{"libsoxr.so.0"};
};

class SoxrResamplerBackend final : public ResamplerBackend
{
public:
  static constexpr const char * kName = "soxr";

  explicit SoxrResamplerBackend(const SoxrResamplerConfig & config);
  ~SoxrResamplerBackend() override;

  SoxrResamplerBackend(const SoxrResamplerBackend &) = delete;
  SoxrResamplerBackend & operator=(const SoxrResamplerBackend &) = delete;

  [[nodiscard]] std::string name() const override;
  [[nodiscard]] std::string quality() const override;
  [[nodiscard]] int targetSampleRate() const override;
  [[nodiscard]] BackendMetrics metrics() const override;
  [[nodiscard]] ProcessResult process(
    const ProcessRequest & request,
    std::vector<uint8_t> & output) override;

private:
  struct Library;
  struct StreamState;

  StreamState makeStreamState(const ProcessRequest & request) const;
  ProcessResult processValidatedFrame(
    StreamState & state,
    const std::vector<float> & samples,
    uint32_t input_frames,
    std::vector<uint8_t> & output);

  SoxrResamplerConfig config_;
  std::unique_ptr<Library> library_;
  mutable std::mutex mutex_;
  std::map<std::string, std::unique_ptr<StreamState>> streams_;
  BackendMetrics metrics_;
};

}  // namespace fa_resample::backends

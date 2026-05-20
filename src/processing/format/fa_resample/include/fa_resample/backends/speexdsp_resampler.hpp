#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "fa_resample/backends/resampler_backend.hpp"

namespace fa_resample::backends
{

struct SpeexDspResamplerConfig
{
  int target_sample_rate{-1};
  int quality{-1};
  std::string library_path{"libspeexdsp.so.1"};
};

class SpeexDspResamplerBackend final : public ResamplerBackend
{
public:
  static constexpr const char * kName = "speexdsp";

  explicit SpeexDspResamplerBackend(const SpeexDspResamplerConfig & config);
  ~SpeexDspResamplerBackend() override;

  SpeexDspResamplerBackend(const SpeexDspResamplerBackend &) = delete;
  SpeexDspResamplerBackend & operator=(const SpeexDspResamplerBackend &) = delete;

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

  SpeexDspResamplerConfig config_;
  std::unique_ptr<Library> library_;
  mutable std::mutex mutex_;
  std::map<std::string, std::unique_ptr<StreamState>> streams_;
  BackendMetrics metrics_;
};

}  // namespace fa_resample::backends

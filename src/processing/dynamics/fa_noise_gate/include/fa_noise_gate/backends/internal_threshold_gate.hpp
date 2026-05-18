#pragma once

#include <cstdint>
#include <vector>

namespace fa_noise_gate::backends
{

struct InternalThresholdGateConfig
{
  int channels{-1};
  double threshold_linear{-1.0};
  double closed_gain_linear{-1.0};
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kNonFiniteInput,
  kOutOfRangeInput,
  kNonFiniteOutput,
  kOutOfRangeOutput,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  uint64_t samples_gated{0};
};

class InternalThresholdGateBackend
{
public:
  static constexpr const char * kName = "internal_threshold_gate";

  explicit InternalThresholdGateBackend(const InternalThresholdGateConfig & config);

  [[nodiscard]] double thresholdLinear() const;
  [[nodiscard]] double closedGainLinear() const;
  [[nodiscard]] int channels() const;

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output) const;

private:
  InternalThresholdGateConfig config_;
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_noise_gate::backends

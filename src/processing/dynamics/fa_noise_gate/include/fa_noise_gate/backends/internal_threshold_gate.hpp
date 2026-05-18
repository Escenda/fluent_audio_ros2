#pragma once

#include <cstdint>
#include <vector>

namespace fa_noise_gate::backends
{

struct InternalThresholdGateConfig
{
  InternalThresholdGateConfig() = delete;
  InternalThresholdGateConfig(
    int channels_value,
    double threshold_linear_value,
    double closed_gain_linear_value);

  int channels;
  double threshold_linear;
  double closed_gain_linear;
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

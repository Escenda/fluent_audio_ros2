#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_normalize::backends
{

struct InternalPeakNormalizeConfig
{
  int channels{-1};
  double target_peak_linear{0.9};
  double silence_threshold_linear{0.0001};
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kNonFiniteInput,
  kOutOfRangeInput,
  kNonFiniteGain,
  kNonFiniteOutput,
  kOutOfRangeOutput,
};

enum class ProcessMode
{
  kNormalized,
  kSilencePassthrough,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  ProcessMode mode{ProcessMode::kNormalized};
  double peak{0.0};
  double gain{1.0};
};

class InternalPeakNormalizeBackend
{
public:
  static constexpr const char * kName = "internal_peak_normalize";

  explicit InternalPeakNormalizeBackend(const InternalPeakNormalizeConfig & config);

  [[nodiscard]] double targetPeakLinear() const;
  [[nodiscard]] double silenceThresholdLinear() const;

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output) const;

private:
  InternalPeakNormalizeConfig config_;
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_normalize::backends

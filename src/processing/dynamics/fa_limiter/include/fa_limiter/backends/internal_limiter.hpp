#pragma once

#include <cstdint>
#include <vector>

namespace fa_limiter::backends
{

struct InternalLimiterConfig
{
  int channels{-1};
  double threshold_linear{-1.0};
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kNonFiniteInput,
  kNonFiniteOutput,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  uint64_t samples_limited{0};
};

class InternalLimiterBackend
{
public:
  static constexpr const char * kName = "internal_limiter";

  explicit InternalLimiterBackend(const InternalLimiterConfig & config);

  [[nodiscard]] double thresholdLinear() const;
  [[nodiscard]] int channels() const;

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output) const;

private:
  InternalLimiterConfig config_;
  float threshold_{1.0F};
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_limiter::backends

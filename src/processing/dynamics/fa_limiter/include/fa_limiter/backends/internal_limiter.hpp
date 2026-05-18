#pragma once

#include <cstdint>
#include <vector>

namespace fa_limiter::backends
{

struct InternalLimiterConfig
{
  InternalLimiterConfig() = delete;
  InternalLimiterConfig(int channels_value, double threshold_linear_value);

  int channels;
  double threshold_linear;
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
  float threshold_;
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_limiter::backends

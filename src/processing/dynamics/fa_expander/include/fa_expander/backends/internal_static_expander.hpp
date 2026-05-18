#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_expander::backends
{

struct InternalStaticExpanderConfig
{
  InternalStaticExpanderConfig() = delete;
  InternalStaticExpanderConfig(
    int channels_value,
    double threshold_linear_value,
    double ratio_value);

  int channels;
  double threshold_linear;
  double ratio;
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
  uint64_t samples_expanded{0};
};

class InternalStaticExpanderBackend
{
public:
  static constexpr const char * kName = "internal_static_expander";

  explicit InternalStaticExpanderBackend(const InternalStaticExpanderConfig & config);

  [[nodiscard]] double thresholdLinear() const;
  [[nodiscard]] double ratio() const;

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output) const;

private:
  InternalStaticExpanderConfig config_;
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_expander::backends

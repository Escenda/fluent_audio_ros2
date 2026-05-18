#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_monitor_mix::backends
{

enum class ProcessStatus
{
  kOk,
  kInputCountMismatch,
  kEmptyInput,
  kMisalignedInput,
  kByteLengthMismatch,
  kNonFiniteInput,
  kOutOfRangeInput,
  kNonFiniteOutput,
  kOutOfRangeOutput,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  std::vector<uint8_t> output;
};

struct InternalMonitorMixConfig
{
  size_t input_count{0};
  size_t master_index{0};
  size_t channels{0};
  std::vector<double> gains_linear{};
};

class InternalMonitorMixBackend
{
public:
  static constexpr const char * kName = "internal_monitor_mix";

  explicit InternalMonitorMixBackend(const InternalMonitorMixConfig & config);

  [[nodiscard]] ProcessStatus validateFrameBytes(const std::vector<uint8_t> & data) const;
  [[nodiscard]] ProcessResult mix(const std::vector<std::vector<uint8_t>> & inputs) const;

private:
  [[nodiscard]] ProcessStatus decodeSamples(
    const std::vector<uint8_t> & data,
    std::vector<float> & samples) const;

  InternalMonitorMixConfig config_;
};

[[nodiscard]] const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_monitor_mix::backends

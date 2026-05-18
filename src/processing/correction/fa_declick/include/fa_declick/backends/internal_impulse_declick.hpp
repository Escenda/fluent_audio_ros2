#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_declick::backends
{

struct InternalImpulseDeclickConfig
{
  int channels{-1};
  double threshold_delta{-1.0};
  int max_click_samples{-1};
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
  uint64_t samples_corrected{0};
  uint64_t click_runs_corrected{0};
};

class InternalImpulseDeclickBackend
{
public:
  static constexpr const char * kName = "internal_impulse_declick";

  explicit InternalImpulseDeclickBackend(const InternalImpulseDeclickConfig & config);

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output) const;

private:
  [[nodiscard]] size_t detectClickRun(
    const std::vector<float> & samples,
    size_t frame_index,
    size_t channel_index,
    size_t frame_count,
    size_t channel_count) const;

  InternalImpulseDeclickConfig config_;
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_declick::backends

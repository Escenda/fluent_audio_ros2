#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_dc_offset_removal::backends
{

struct InternalFrameMeanConfig
{
  int channels{-1};
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kNonFiniteInput,
  kNonFiniteMean,
  kNonFiniteOutput,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
};

class InternalFrameMeanBackend
{
public:
  static constexpr const char * kName = "internal_frame_mean";

  explicit InternalFrameMeanBackend(const InternalFrameMeanConfig & config);

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output) const;

private:
  InternalFrameMeanConfig config_;
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_dc_offset_removal::backends

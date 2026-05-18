#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_trim::backends
{

struct InternalFrameTrimConfig
{
  int channels{0};
  size_t leading_frames{0U};
  size_t trailing_frames{0U};
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kTrimExhaustsInput,
  kNonFiniteInput,
  kOutOfRangeInput,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  size_t input_frame_count{0U};
  size_t output_frame_count{0U};
};

class InternalFrameTrimBackend
{
public:
  static constexpr const char * kName = "internal_frame_trim";

  explicit InternalFrameTrimBackend(const InternalFrameTrimConfig & config);

  [[nodiscard]] int channels() const;
  [[nodiscard]] size_t leadingFrames() const;
  [[nodiscard]] size_t trailingFrames() const;

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output) const;

private:
  InternalFrameTrimConfig config_;
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_trim::backends

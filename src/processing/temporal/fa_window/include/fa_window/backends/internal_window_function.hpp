#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_window::backends
{

enum class WindowType
{
  kHann,
  kHamming,
};

struct InternalWindowFunctionConfig
{
  int channels{0};
  WindowType window_type{WindowType::kHann};
  size_t expected_frames{0U};
  bool strict_frame_count{true};
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kFrameCountMismatch,
  kTooFewFrames,
  kNonFiniteInput,
  kOutOfRangeInput,
  kNonFiniteOutput,
  kOutOfRangeOutput,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  size_t input_frame_count{0U};
  size_t output_frame_count{0U};
};

class InternalWindowFunctionBackend
{
public:
  static constexpr const char * kName = "internal_window_function";

  explicit InternalWindowFunctionBackend(const InternalWindowFunctionConfig & config);

  [[nodiscard]] int channels() const;
  [[nodiscard]] WindowType windowType() const;
  [[nodiscard]] size_t expectedFrames() const;
  [[nodiscard]] bool strictFrameCount() const;

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output) const;

private:
  [[nodiscard]] double coefficientAt(size_t frame_index, size_t frame_count) const;

  InternalWindowFunctionConfig config_;
};

const char * windowTypeName(WindowType window_type);
const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_window::backends

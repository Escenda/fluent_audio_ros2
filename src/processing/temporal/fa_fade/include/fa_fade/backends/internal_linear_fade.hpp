#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_fade::backends
{

enum class FadeMode
{
  kFadeIn,
  kFadeOut,
};

struct InternalLinearFadeConfig
{
  int channels{0};
  FadeMode mode{FadeMode::kFadeIn};
  size_t duration_frames{0U};
  uint64_t initial_position_frames{0U};
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kPositionOverflow,
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
  uint64_t position_frames{0U};
};

class InternalLinearFadeBackend
{
public:
  static constexpr const char * kName = "internal_linear_fade";

  explicit InternalLinearFadeBackend(const InternalLinearFadeConfig & config);

  [[nodiscard]] int channels() const;
  [[nodiscard]] FadeMode mode() const;
  [[nodiscard]] size_t durationFrames() const;
  [[nodiscard]] uint64_t positionFrames() const;

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output);

private:
  [[nodiscard]] double gainAtPosition(uint64_t position_frames) const;

  InternalLinearFadeConfig config_;
  uint64_t position_frames_{0U};
};

const char * fadeModeName(FadeMode mode);
const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_fade::backends

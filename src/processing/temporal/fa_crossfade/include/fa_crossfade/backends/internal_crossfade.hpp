#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_crossfade::backends
{

enum class FadeCurve
{
  kLinear,
  kEqualPower,
};

struct InternalCrossfadeConfig
{
  int channels{0};
  size_t overlap_frames{0U};
  FadeCurve fade_curve{FadeCurve::kLinear};
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kInputTooShort,
  kNonFiniteInput,
  kOutOfRangeInput,
  kNonFiniteOutput,
  kOutOfRangeOutput,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  size_t output_frames{0U};
};

class InternalCrossfadeBackend
{
public:
  static constexpr const char * kName = "internal_crossfade";

  explicit InternalCrossfadeBackend(const InternalCrossfadeConfig & config);

  [[nodiscard]] int channels() const;
  [[nodiscard]] size_t overlapFrames() const;
  [[nodiscard]] FadeCurve fadeCurve() const;

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & segment_a,
    const std::vector<uint8_t> & segment_b,
    std::vector<uint8_t> & output) const;

private:
  InternalCrossfadeConfig config_;
};

FadeCurve fadeCurveFromName(const std::string & name);
const char * fadeCurveName(FadeCurve curve);
const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_crossfade::backends

#pragma once

#include <cstdint>
#include <vector>

namespace fa_compressor::backends
{

struct InternalStaticCurveConfig
{
  int channels{-1};
  double threshold_linear{0.5};
  double ratio{4.0};
  double makeup_gain_linear{1.0};
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
  uint64_t samples_compressed{0};
};

class InternalStaticCurveBackend
{
public:
  static constexpr const char * kName = "internal_static_curve";

  explicit InternalStaticCurveBackend(const InternalStaticCurveConfig & config);

  [[nodiscard]] double thresholdLinear() const;
  [[nodiscard]] double ratio() const;
  [[nodiscard]] double makeupGainLinear() const;
  [[nodiscard]] int channels() const;

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output) const;

private:
  InternalStaticCurveConfig config_;
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_compressor::backends

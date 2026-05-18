#pragma once

#include <cstdint>
#include <vector>

namespace fa_gain::backends
{

struct InternalGainConfig
{
  int channels{-1};
  double linear_gain{-1.0};
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

class InternalGainBackend
{
public:
  static constexpr const char * kName = "internal_gain";

  explicit InternalGainBackend(const InternalGainConfig & config);

  [[nodiscard]] double linearGain() const;
  [[nodiscard]] int channels() const;

  [[nodiscard]] ProcessStatus process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output) const;

private:
  InternalGainConfig config_;
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_gain::backends

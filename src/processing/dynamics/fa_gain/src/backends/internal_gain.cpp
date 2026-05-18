#include "fa_gain/backends/internal_gain.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fa_gain::backends
{

namespace
{
constexpr float kNormalizedMin = -1.0F;
constexpr float kNormalizedMax = 1.0F;

bool isFinite(double value)
{
  return std::isfinite(value);
}

bool isNormalizedSample(float value)
{
  return std::isfinite(value) && value >= kNormalizedMin && value <= kNormalizedMax;
}
}  // namespace

InternalGainBackend::InternalGainBackend(const InternalGainConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (!isFinite(config_.linear_gain) || config_.linear_gain < 0.0) {
    throw std::runtime_error("gain.linear must be finite and >= 0.0");
  }
}

double InternalGainBackend::linearGain() const
{
  return config_.linear_gain;
}

int InternalGainBackend::channels() const
{
  return config_.channels;
}

ProcessStatus InternalGainBackend::process(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output) const
{
  const size_t bytes_per_frame = static_cast<size_t>(config_.channels) * sizeof(float);
  if (input.empty()) {
    return ProcessStatus::kEmptyInput;
  }
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessStatus::kMisalignedInput;
  }

  std::vector<uint8_t> next_output(input.size());
  const double float_max = static_cast<double>(std::numeric_limits<float>::max());
  const size_t sample_count = input.size() / sizeof(float);

  for (size_t i = 0; i < sample_count; ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, input.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample)) {
      return ProcessStatus::kNonFiniteInput;
    }
    if (!isNormalizedSample(sample)) {
      return ProcessStatus::kOutOfRangeInput;
    }

    const double gained = static_cast<double>(sample) * config_.linear_gain;
    if (!isFinite(gained) || gained > float_max || gained < -float_max) {
      return ProcessStatus::kNonFiniteOutput;
    }
    if (gained < kNormalizedMin || gained > kNormalizedMax) {
      return ProcessStatus::kOutOfRangeOutput;
    }

    const float out_sample = static_cast<float>(gained);
    if (!isNormalizedSample(out_sample)) {
      return ProcessStatus::kOutOfRangeOutput;
    }
    std::memcpy(next_output.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  output = std::move(next_output);
  return ProcessStatus::kOk;
}

const char * processStatusMessage(ProcessStatus status)
{
  switch (status) {
    case ProcessStatus::kOk:
      return "ok";
    case ProcessStatus::kEmptyInput:
      return "input data is empty";
    case ProcessStatus::kMisalignedInput:
      return "input byte length is not aligned to FLOAT32LE interleaved frames";
    case ProcessStatus::kNonFiniteInput:
      return "input sample is not finite";
    case ProcessStatus::kOutOfRangeInput:
      return "input sample is outside normalized FLOAT32LE range";
    case ProcessStatus::kNonFiniteOutput:
      return "gain output sample is not representable as finite FLOAT32LE";
    case ProcessStatus::kOutOfRangeOutput:
      return "gain output sample is outside normalized FLOAT32LE range";
  }
  return "unknown gain backend status";
}

}  // namespace fa_gain::backends

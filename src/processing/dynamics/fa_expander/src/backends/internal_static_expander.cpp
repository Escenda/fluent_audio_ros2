#include "fa_expander/backends/internal_static_expander.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_expander::backends
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

InternalStaticExpanderConfig::InternalStaticExpanderConfig(
  int channels_value,
  double threshold_linear_value,
  double ratio_value)
: channels(channels_value),
  threshold_linear(threshold_linear_value),
  ratio(ratio_value)
{
}

InternalStaticExpanderBackend::InternalStaticExpanderBackend(
  const InternalStaticExpanderConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (!isFinite(config_.threshold_linear) ||
      config_.threshold_linear <= 0.0 ||
      config_.threshold_linear >= 1.0)
  {
    throw std::runtime_error("expander.threshold_linear must be finite and in (0.0, 1.0)");
  }
  if (!isFinite(config_.ratio) || config_.ratio <= 1.0) {
    throw std::runtime_error("expander.ratio must be finite and > 1.0");
  }
}

double InternalStaticExpanderBackend::thresholdLinear() const
{
  return config_.threshold_linear;
}

double InternalStaticExpanderBackend::ratio() const
{
  return config_.ratio;
}

ProcessResult InternalStaticExpanderBackend::process(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output) const
{
  const size_t bytes_per_frame = static_cast<size_t>(config_.channels) * sizeof(float);
  if (input.empty()) {
    return ProcessResult{ProcessStatus::kEmptyInput, 0};
  }
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessResult{ProcessStatus::kMisalignedInput, 0};
  }

  std::vector<uint8_t> next_output(input.size());
  uint64_t samples_expanded = 0;
  for (size_t i = 0; i < input.size() / sizeof(float); ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, input.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteInput, 0};
    }
    if (!isNormalizedSample(sample)) {
      return ProcessResult{ProcessStatus::kOutOfRangeInput, 0};
    }

    const double magnitude = std::abs(static_cast<double>(sample));
    double expanded = static_cast<double>(sample);
    if (magnitude < config_.threshold_linear) {
      const double expanded_abs =
        config_.threshold_linear *
        std::pow(magnitude / config_.threshold_linear, config_.ratio);
      expanded = std::copysign(expanded_abs, static_cast<double>(sample));
      ++samples_expanded;
    }

    if (!isFinite(expanded)) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput, 0};
    }
    if (expanded < kNormalizedMin || expanded > kNormalizedMax) {
      return ProcessResult{ProcessStatus::kOutOfRangeOutput, 0};
    }

    const float out_sample = static_cast<float>(expanded);
    if (!isNormalizedSample(out_sample)) {
      return ProcessResult{ProcessStatus::kOutOfRangeOutput, 0};
    }
    std::memcpy(next_output.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  output = std::move(next_output);
  return ProcessResult{ProcessStatus::kOk, samples_expanded};
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
      return "expander output sample is not representable as finite FLOAT32LE";
    case ProcessStatus::kOutOfRangeOutput:
      return "expander output sample is outside normalized FLOAT32LE range";
  }
  throw std::logic_error("unhandled expander backend process status");
}

}  // namespace fa_expander::backends

#include "fa_noise_gate/backends/internal_threshold_gate.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_noise_gate::backends
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

InternalThresholdGateConfig::InternalThresholdGateConfig(
  int channels_value,
  double threshold_linear_value,
  double closed_gain_linear_value)
: channels(channels_value),
  threshold_linear(threshold_linear_value),
  closed_gain_linear(closed_gain_linear_value)
{
}

InternalThresholdGateBackend::InternalThresholdGateBackend(
  const InternalThresholdGateConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (!isFinite(config_.threshold_linear) ||
      config_.threshold_linear < 0.0 ||
      config_.threshold_linear > 1.0)
  {
    throw std::runtime_error("gate.threshold_linear must be finite and in [0.0, 1.0]");
  }
  if (!isFinite(config_.closed_gain_linear) ||
      config_.closed_gain_linear < 0.0 ||
      config_.closed_gain_linear > 1.0)
  {
    throw std::runtime_error("gate.closed_gain_linear must be finite and in [0.0, 1.0]");
  }
}

double InternalThresholdGateBackend::thresholdLinear() const
{
  return config_.threshold_linear;
}

double InternalThresholdGateBackend::closedGainLinear() const
{
  return config_.closed_gain_linear;
}

int InternalThresholdGateBackend::channels() const
{
  return config_.channels;
}

ProcessResult InternalThresholdGateBackend::process(
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
  uint64_t gated_in_frame = 0;
  const size_t sample_count = input.size() / sizeof(float);

  for (size_t i = 0; i < sample_count; ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, input.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteInput, 0};
    }
    if (!isNormalizedSample(sample)) {
      return ProcessResult{ProcessStatus::kOutOfRangeInput, 0};
    }

    double output_sample = static_cast<double>(sample);
    if (std::abs(sample) < config_.threshold_linear) {
      output_sample = static_cast<double>(sample) * config_.closed_gain_linear;
      ++gated_in_frame;
    }
    if (!isFinite(output_sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput, 0};
    }
    if (output_sample < kNormalizedMin || output_sample > kNormalizedMax) {
      return ProcessResult{ProcessStatus::kOutOfRangeOutput, 0};
    }

    const float out_sample = static_cast<float>(output_sample);
    if (!isNormalizedSample(out_sample)) {
      return ProcessResult{ProcessStatus::kOutOfRangeOutput, 0};
    }
    std::memcpy(next_output.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  output = std::move(next_output);
  return ProcessResult{ProcessStatus::kOk, gated_in_frame};
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
      return "noise gate output sample is not representable as finite FLOAT32LE";
    case ProcessStatus::kOutOfRangeOutput:
      return "noise gate output sample is outside normalized FLOAT32LE range";
  }
  return "unknown noise gate backend status";
}

}  // namespace fa_noise_gate::backends

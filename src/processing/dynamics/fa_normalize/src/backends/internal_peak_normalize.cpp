#include "fa_normalize/backends/internal_peak_normalize.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_normalize::backends
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

InternalPeakNormalizeBackend::InternalPeakNormalizeBackend(
  const InternalPeakNormalizeConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (!isFinite(config_.target_peak_linear) ||
      config_.target_peak_linear <= 0.0 ||
      config_.target_peak_linear > 1.0)
  {
    throw std::runtime_error("normalize.target_peak_linear must be finite and in (0.0, 1.0]");
  }
  if (!isFinite(config_.silence_threshold_linear) ||
      config_.silence_threshold_linear < 0.0 ||
      config_.silence_threshold_linear >= config_.target_peak_linear)
  {
    throw std::runtime_error(
      "normalize.silence_threshold_linear must be finite, >= 0.0, and < normalize.target_peak_linear");
  }
}

double InternalPeakNormalizeBackend::targetPeakLinear() const
{
  return config_.target_peak_linear;
}

double InternalPeakNormalizeBackend::silenceThresholdLinear() const
{
  return config_.silence_threshold_linear;
}

ProcessResult InternalPeakNormalizeBackend::process(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output) const
{
  const size_t bytes_per_frame = static_cast<size_t>(config_.channels) * sizeof(float);
  if (input.empty()) {
    return ProcessResult{ProcessStatus::kEmptyInput, ProcessMode::kNormalized, 0.0, 1.0};
  }
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessResult{ProcessStatus::kMisalignedInput, ProcessMode::kNormalized, 0.0, 1.0};
  }

  std::vector<float> samples(input.size() / sizeof(float));
  float peak = 0.0F;
  for (size_t i = 0; i < samples.size(); ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, input.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteInput, ProcessMode::kNormalized, 0.0, 1.0};
    }
    if (!isNormalizedSample(sample)) {
      return ProcessResult{ProcessStatus::kOutOfRangeInput, ProcessMode::kNormalized, 0.0, 1.0};
    }
    samples[i] = sample;
    peak = std::max(peak, std::abs(sample));
  }

  if (peak < static_cast<float>(config_.silence_threshold_linear)) {
    output = input;
    return ProcessResult{
      ProcessStatus::kOk,
      ProcessMode::kSilencePassthrough,
      static_cast<double>(peak),
      1.0};
  }

  const double gain = config_.target_peak_linear / static_cast<double>(peak);
  if (!isFinite(gain)) {
    return ProcessResult{
      ProcessStatus::kNonFiniteGain,
      ProcessMode::kNormalized,
      static_cast<double>(peak),
      1.0};
  }

  std::vector<uint8_t> next_output(input.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    const double normalized = static_cast<double>(samples[i]) * gain;
    if (!isFinite(normalized)) {
      return ProcessResult{
        ProcessStatus::kNonFiniteOutput,
        ProcessMode::kNormalized,
        static_cast<double>(peak),
        1.0};
    }
    if (normalized < kNormalizedMin || normalized > kNormalizedMax) {
      return ProcessResult{
        ProcessStatus::kOutOfRangeOutput,
        ProcessMode::kNormalized,
        static_cast<double>(peak),
        1.0};
    }

    const float out_sample = static_cast<float>(normalized);
    if (!isNormalizedSample(out_sample)) {
      return ProcessResult{
        ProcessStatus::kOutOfRangeOutput,
        ProcessMode::kNormalized,
        static_cast<double>(peak),
        1.0};
    }
    std::memcpy(next_output.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  output = std::move(next_output);
  return ProcessResult{
    ProcessStatus::kOk,
    ProcessMode::kNormalized,
    static_cast<double>(peak),
    gain};
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
    case ProcessStatus::kNonFiniteGain:
      return "normalize gain is not finite";
    case ProcessStatus::kNonFiniteOutput:
      return "normalize output sample is not representable as finite FLOAT32LE";
    case ProcessStatus::kOutOfRangeOutput:
      return "normalize output sample is outside normalized FLOAT32LE range";
  }
  return "unknown normalize backend status";
}

}  // namespace fa_normalize::backends

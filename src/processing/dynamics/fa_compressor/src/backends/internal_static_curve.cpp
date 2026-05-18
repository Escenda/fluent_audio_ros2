#include "fa_compressor/backends/internal_static_curve.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_compressor::backends
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

InternalStaticCurveBackend::InternalStaticCurveBackend(
  const InternalStaticCurveConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (!isFinite(config_.threshold_linear) ||
      config_.threshold_linear <= 0.0 ||
      config_.threshold_linear >= 1.0)
  {
    throw std::runtime_error("compressor.threshold_linear must be finite and in (0.0, 1.0)");
  }
  if (!isFinite(config_.ratio) || config_.ratio <= 1.0) {
    throw std::runtime_error("compressor.ratio must be finite and > 1.0");
  }
  if (!isFinite(config_.makeup_gain_linear) ||
      config_.makeup_gain_linear <= 0.0 ||
      config_.makeup_gain_linear > 4.0)
  {
    throw std::runtime_error("compressor.makeup_gain_linear must be finite and in (0.0, 4.0]");
  }
}

double InternalStaticCurveBackend::thresholdLinear() const
{
  return config_.threshold_linear;
}

double InternalStaticCurveBackend::ratio() const
{
  return config_.ratio;
}

double InternalStaticCurveBackend::makeupGainLinear() const
{
  return config_.makeup_gain_linear;
}

int InternalStaticCurveBackend::channels() const
{
  return config_.channels;
}

ProcessResult InternalStaticCurveBackend::process(
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
  uint64_t compressed_in_frame = 0;
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

    const double amplitude = std::abs(static_cast<double>(sample));
    double compressed_abs = amplitude;
    if (amplitude > config_.threshold_linear) {
      compressed_abs =
        config_.threshold_linear + ((amplitude - config_.threshold_linear) / config_.ratio);
      ++compressed_in_frame;
    }

    const double signed_sample = std::signbit(sample) ? -compressed_abs : compressed_abs;
    const double output_sample = signed_sample * config_.makeup_gain_linear;
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
  return ProcessResult{ProcessStatus::kOk, compressed_in_frame};
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
      return "compressor output sample is not representable as finite FLOAT32LE";
    case ProcessStatus::kOutOfRangeOutput:
      return "compressor output sample is outside normalized FLOAT32LE range";
  }
  return "unknown compressor backend status";
}

}  // namespace fa_compressor::backends

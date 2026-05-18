#include "fa_limiter/backends/internal_limiter.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_limiter::backends
{

namespace
{
bool isFinite(double value)
{
  return std::isfinite(value);
}
}  // namespace

InternalLimiterBackend::InternalLimiterBackend(const InternalLimiterConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (!isFinite(config_.threshold_linear) ||
      config_.threshold_linear <= 0.0 ||
      config_.threshold_linear > 1.0)
  {
    throw std::runtime_error("threshold.linear must be finite and in (0.0, 1.0]");
  }

  threshold_ = static_cast<float>(config_.threshold_linear);
  if (!std::isfinite(threshold_) || threshold_ <= 0.0F || threshold_ > 1.0F) {
    throw std::runtime_error("threshold.linear cannot be represented as valid FLOAT32LE");
  }
}

double InternalLimiterBackend::thresholdLinear() const
{
  return config_.threshold_linear;
}

int InternalLimiterBackend::channels() const
{
  return config_.channels;
}

ProcessResult InternalLimiterBackend::process(
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
  uint64_t limited_in_frame = 0;
  const size_t sample_count = input.size() / sizeof(float);

  for (size_t i = 0; i < sample_count; ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, input.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteInput, 0};
    }

    float out_sample = sample;
    if (sample > threshold_) {
      out_sample = threshold_;
      ++limited_in_frame;
    } else if (sample < -threshold_) {
      out_sample = -threshold_;
      ++limited_in_frame;
    }
    if (!std::isfinite(out_sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput, 0};
    }
    std::memcpy(next_output.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  output = std::move(next_output);
  return ProcessResult{ProcessStatus::kOk, limited_in_frame};
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
    case ProcessStatus::kNonFiniteOutput:
      return "limiter output sample is not finite";
  }
  return "unknown limiter backend status";
}

}  // namespace fa_limiter::backends

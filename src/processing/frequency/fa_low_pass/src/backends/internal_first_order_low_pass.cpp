#include "fa_low_pass/backends/internal_first_order_low_pass.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_low_pass::backends
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr float kNormalizedMin = -1.0F;
constexpr float kNormalizedMax = 1.0F;

bool isNormalizedSample(float value)
{
  return std::isfinite(value) && value >= kNormalizedMin && value <= kNormalizedMax;
}
}  // namespace

InternalFirstOrderLowPassBackend::InternalFirstOrderLowPassBackend(
  const InternalFirstOrderLowPassConfig & config)
: config_(config)
{
  if (config_.sample_rate <= 0) {
    throw std::runtime_error("feature.sample_rate must be > 0");
  }
  if (config_.channels <= 0) {
    throw std::runtime_error("feature.channels must be > 0");
  }
  const double nyquist_hz = static_cast<double>(config_.sample_rate) / 2.0;
  if (!std::isfinite(config_.cutoff_hz) || config_.cutoff_hz <= 0.0 ||
      config_.cutoff_hz >= nyquist_hz)
  {
    throw std::runtime_error(
      "filter.cutoff_hz must be finite, > 0.0, and < sample_rate / 2.0");
  }

  const double sample_interval_sec = 1.0 / static_cast<double>(config_.sample_rate);
  const double rc_sec = 1.0 / (2.0 * kPi * config_.cutoff_hz);
  alpha_ = sample_interval_sec / (rc_sec + sample_interval_sec);
  if (!std::isfinite(alpha_) || alpha_ <= 0.0 || alpha_ >= 1.0) {
    throw std::runtime_error("low-pass coefficient alpha must be finite and in (0.0, 1.0)");
  }
  channel_states_.assign(static_cast<size_t>(config_.channels), ChannelFilterState{});
}

double InternalFirstOrderLowPassBackend::alpha() const
{
  return alpha_;
}

int InternalFirstOrderLowPassBackend::channels() const
{
  return config_.channels;
}

double InternalFirstOrderLowPassBackend::cutoffHz() const
{
  return config_.cutoff_hz;
}

ProcessStatus InternalFirstOrderLowPassBackend::process(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output,
  bool reset_state)
{
  const size_t bytes_per_frame = static_cast<size_t>(config_.channels) * sizeof(float);
  if (input.empty()) {
    return ProcessStatus::kEmptyInput;
  }
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessStatus::kMisalignedInput;
  }

  std::vector<ChannelFilterState> next_channel_states =
    reset_state ?
    std::vector<ChannelFilterState>(
      static_cast<size_t>(config_.channels), ChannelFilterState{}) :
    channel_states_;
  std::vector<uint8_t> next_output(input.size());

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

    ChannelFilterState & state =
      next_channel_states.at(i % static_cast<size_t>(config_.channels));
    float out_sample = sample;
    if (state.initialized) {
      const double filtered =
        static_cast<double>(state.previous_output) +
        alpha_ * (static_cast<double>(sample) - static_cast<double>(state.previous_output));
      if (!std::isfinite(filtered)) {
        return ProcessStatus::kNonFiniteOutput;
      }
      if (filtered < kNormalizedMin || filtered > kNormalizedMax) {
        return ProcessStatus::kOutOfRangeOutput;
      }
      out_sample = static_cast<float>(filtered);
    }

    state.previous_output = out_sample;
    state.initialized = true;
    std::memcpy(next_output.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  channel_states_ = std::move(next_channel_states);
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
      return "low-pass output sample is not finite";
    case ProcessStatus::kOutOfRangeOutput:
      return "low-pass output sample is outside normalized FLOAT32LE range";
  }
  throw std::logic_error("unhandled low-pass backend process status");
}

}  // namespace fa_low_pass::backends

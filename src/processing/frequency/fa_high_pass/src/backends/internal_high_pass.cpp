#include "fa_high_pass/backends/internal_high_pass.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fa_high_pass::backends
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
}

InternalHighPassBackend::InternalHighPassBackend(const InternalHighPassConfig & config)
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
  alpha_ = rc_sec / (rc_sec + sample_interval_sec);
  channel_states_.assign(static_cast<size_t>(config_.channels), ChannelFilterState{});
}

double InternalHighPassBackend::alpha() const
{
  return alpha_;
}

int InternalHighPassBackend::channels() const
{
  return config_.channels;
}

double InternalHighPassBackend::cutoffHz() const
{
  return config_.cutoff_hz;
}

ProcessStatus InternalHighPassBackend::process(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output)
{
  const size_t bytes_per_frame = static_cast<size_t>(config_.channels) * sizeof(float);
  if (input.empty()) {
    return ProcessStatus::kEmptyInput;
  }
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessStatus::kMisalignedInput;
  }

  std::vector<ChannelFilterState> next_channel_states = channel_states_;
  std::vector<uint8_t> next_output(input.size());
  const size_t sample_count = input.size() / sizeof(float);
  for (size_t i = 0; i < sample_count; ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, input.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample)) {
      return ProcessStatus::kNonFiniteInput;
    }

    ChannelFilterState & state =
      next_channel_states.at(i % static_cast<size_t>(config_.channels));
    float out_sample = 0.0F;
    if (state.initialized) {
      const double filtered =
        alpha_ * (static_cast<double>(state.previous_output) +
        static_cast<double>(sample) - static_cast<double>(state.previous_input));
      const double float_max = static_cast<double>(std::numeric_limits<float>::max());
      if (!std::isfinite(filtered) || filtered > float_max || filtered < -float_max) {
        return ProcessStatus::kNonFiniteOutput;
      }
      out_sample = static_cast<float>(filtered);
    }

    state.previous_input = sample;
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
    case ProcessStatus::kNonFiniteOutput:
      return "high-pass output sample is not representable as finite FLOAT32LE";
  }
  throw std::logic_error("unhandled high-pass backend process status");
}

}  // namespace fa_high_pass::backends

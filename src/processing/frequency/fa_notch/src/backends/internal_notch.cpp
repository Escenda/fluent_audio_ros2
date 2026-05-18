#include "fa_notch/backends/internal_notch.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fa_notch::backends
{

namespace
{
constexpr double kPi = 3.14159265358979323846;

bool isFinite(double value)
{
  return std::isfinite(value);
}
}  // namespace

InternalNotchBackend::InternalNotchBackend(const InternalNotchConfig & config)
: config_(config)
{
  if (config_.sample_rate <= 0) {
    throw std::runtime_error("feature.sample_rate must be > 0");
  }
  if (config_.channels <= 0) {
    throw std::runtime_error("feature.channels must be > 0");
  }
  const double nyquist_hz = static_cast<double>(config_.sample_rate) / 2.0;
  if (!isFinite(config_.center_hz) || config_.center_hz <= 0.0 ||
      config_.center_hz >= nyquist_hz)
  {
    throw std::runtime_error(
      "filter.center_hz must be finite, > 0.0, and < sample_rate / 2.0");
  }
  if (!isFinite(config_.q) || config_.q <= 0.0) {
    throw std::runtime_error("filter.q must be finite and > 0.0");
  }

  const double omega = 2.0 * kPi * config_.center_hz / static_cast<double>(config_.sample_rate);
  const double alpha = std::sin(omega) / (2.0 * config_.q);
  const double cos_omega = std::cos(omega);
  const double a0 = 1.0 + alpha;
  if (!isFinite(a0) || a0 == 0.0) {
    throw std::runtime_error("notch coefficient normalization failed because a0 is invalid");
  }

  coefficients_.b0 = 1.0 / a0;
  coefficients_.b1 = (-2.0 * cos_omega) / a0;
  coefficients_.b2 = 1.0 / a0;
  coefficients_.a1 = (-2.0 * cos_omega) / a0;
  coefficients_.a2 = (1.0 - alpha) / a0;

  if (!isFinite(coefficients_.b0) || !isFinite(coefficients_.b1) ||
      !isFinite(coefficients_.b2) || !isFinite(coefficients_.a1) ||
      !isFinite(coefficients_.a2))
  {
    throw std::runtime_error("notch coefficient normalization produced non-finite coefficients");
  }

  channel_states_.assign(static_cast<size_t>(config_.channels), ChannelFilterState{});
}

double InternalNotchBackend::centerHz() const
{
  return config_.center_hz;
}

double InternalNotchBackend::q() const
{
  return config_.q;
}

int InternalNotchBackend::channels() const
{
  return config_.channels;
}

const BiquadCoefficients & InternalNotchBackend::coefficients() const
{
  return coefficients_;
}

ProcessStatus InternalNotchBackend::process(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output,
  const bool reset_state)
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
    std::vector<ChannelFilterState>(channel_states_.size(), ChannelFilterState{}) :
    channel_states_;
  std::vector<uint8_t> next_output(input.size());
  const double float_max = static_cast<double>(std::numeric_limits<float>::max());

  const size_t sample_count = input.size() / sizeof(float);
  for (size_t i = 0; i < sample_count; ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, input.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample)) {
      return ProcessStatus::kNonFiniteInput;
    }

    ChannelFilterState & state =
      next_channel_states.at(i % static_cast<size_t>(config_.channels));
    const double input_sample = static_cast<double>(sample);
    const double filtered =
      coefficients_.b0 * input_sample +
      coefficients_.b1 * state.previous_input_1 +
      coefficients_.b2 * state.previous_input_2 -
      coefficients_.a1 * state.previous_output_1 -
      coefficients_.a2 * state.previous_output_2;
    if (!isFinite(filtered) || filtered > float_max || filtered < -float_max) {
      return ProcessStatus::kNonFiniteOutput;
    }

    const float out_sample = static_cast<float>(filtered);
    state.previous_input_2 = state.previous_input_1;
    state.previous_input_1 = input_sample;
    state.previous_output_2 = state.previous_output_1;
    state.previous_output_1 = filtered;
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
      return "notch output sample is not representable as finite FLOAT32LE";
  }
  return "unknown notch backend status";
}

}  // namespace fa_notch::backends

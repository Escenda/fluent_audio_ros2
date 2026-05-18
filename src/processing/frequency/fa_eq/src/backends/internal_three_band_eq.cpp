#include "fa_eq/backends/internal_three_band_eq.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fa_eq::backends
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
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

double dbToLinear(double gain_db)
{
  return std::pow(10.0, gain_db / 20.0);
}
}  // namespace

InternalThreeBandEqBackend::InternalThreeBandEqBackend(
  const InternalThreeBandEqConfig & config)
: config_(config)
{
  if (config_.sample_rate <= 0) {
    throw std::runtime_error("feature.sample_rate must be > 0");
  }
  if (config_.channels <= 0) {
    throw std::runtime_error("feature.channels must be > 0");
  }
  const double nyquist_hz = static_cast<double>(config_.sample_rate) / 2.0;
  if (!isFinite(config_.low_cutoff_hz) || config_.low_cutoff_hz <= 0.0) {
    throw std::runtime_error("low.cutoff_hz must be finite and > 0.0");
  }
  if (!isFinite(config_.high_cutoff_hz) ||
      config_.high_cutoff_hz <= config_.low_cutoff_hz ||
      config_.high_cutoff_hz >= nyquist_hz)
  {
    throw std::runtime_error(
      "high.cutoff_hz must be finite, > low.cutoff_hz, and < sample_rate / 2.0");
  }
  if (!isFinite(config_.gain_low_db) ||
      !isFinite(config_.gain_mid_db) ||
      !isFinite(config_.gain_high_db))
  {
    throw std::runtime_error("gains.*_db must be finite");
  }

  const double dt = 1.0 / static_cast<double>(config_.sample_rate);
  const double low_rc = 1.0 / (2.0 * kPi * config_.low_cutoff_hz);
  const double high_rc = 1.0 / (2.0 * kPi * config_.high_cutoff_hz);
  low_alpha_ = dt / (low_rc + dt);
  high_alpha_ = high_rc / (high_rc + dt);
  if (!isFinite(low_alpha_) || low_alpha_ <= 0.0 || low_alpha_ >= 1.0) {
    throw std::runtime_error("low split coefficient alpha must be finite and in (0.0, 1.0)");
  }
  if (!isFinite(high_alpha_) || high_alpha_ <= 0.0 || high_alpha_ >= 1.0) {
    throw std::runtime_error("high split coefficient alpha must be finite and in (0.0, 1.0)");
  }

  gain_low_linear_ = dbToLinear(config_.gain_low_db);
  gain_mid_linear_ = dbToLinear(config_.gain_mid_db);
  gain_high_linear_ = dbToLinear(config_.gain_high_db);
  if (!isFinite(gain_low_linear_) ||
      !isFinite(gain_mid_linear_) ||
      !isFinite(gain_high_linear_) ||
      gain_low_linear_ <= 0.0 ||
      gain_mid_linear_ <= 0.0 ||
      gain_high_linear_ <= 0.0)
  {
    throw std::runtime_error("linear gains derived from gains.*_db must be finite and > 0.0");
  }

  channel_states_.assign(static_cast<size_t>(config_.channels), ChannelFilterState{});
}

double InternalThreeBandEqBackend::lowAlpha() const
{
  return low_alpha_;
}

double InternalThreeBandEqBackend::highAlpha() const
{
  return high_alpha_;
}

double InternalThreeBandEqBackend::gainLowLinear() const
{
  return gain_low_linear_;
}

double InternalThreeBandEqBackend::gainMidLinear() const
{
  return gain_mid_linear_;
}

double InternalThreeBandEqBackend::gainHighLinear() const
{
  return gain_high_linear_;
}

int InternalThreeBandEqBackend::channels() const
{
  return config_.channels;
}

ProcessStatus InternalThreeBandEqBackend::process(
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

    ChannelFilterState & state =
      next_channel_states.at(i % static_cast<size_t>(config_.channels));
    float low_sample = sample;
    float high_sample = 0.0F;
    if (state.initialized) {
      const double low_split =
        static_cast<double>(state.previous_low_output) +
        low_alpha_ * (static_cast<double>(sample) -
        static_cast<double>(state.previous_low_output));
      if (!isFinite(low_split) || low_split > float_max || low_split < -float_max) {
        return ProcessStatus::kNonFiniteOutput;
      }
      low_sample = static_cast<float>(low_split);

      const double high_split =
        high_alpha_ * (static_cast<double>(state.previous_hp_output) +
        static_cast<double>(sample) - static_cast<double>(state.previous_hp_input));
      if (!isFinite(high_split) || high_split > float_max || high_split < -float_max) {
        return ProcessStatus::kNonFiniteOutput;
      }
      high_sample = static_cast<float>(high_split);
    }

    const double mid_sample =
      static_cast<double>(sample) - static_cast<double>(low_sample) -
      static_cast<double>(high_sample);
    if (!isFinite(mid_sample)) {
      return ProcessStatus::kNonFiniteOutput;
    }

    const double mixed =
      (static_cast<double>(low_sample) * gain_low_linear_) +
      (mid_sample * gain_mid_linear_) +
      (static_cast<double>(high_sample) * gain_high_linear_);
    if (!isFinite(mixed) || mixed > float_max || mixed < -float_max) {
      return ProcessStatus::kNonFiniteOutput;
    }
    if (mixed < kNormalizedMin || mixed > kNormalizedMax) {
      return ProcessStatus::kOutOfRangeOutput;
    }

    const float out_sample = static_cast<float>(mixed);
    state.previous_low_output = low_sample;
    state.previous_hp_input = sample;
    state.previous_hp_output = high_sample;
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
      return "EQ output sample is not representable as finite FLOAT32LE";
    case ProcessStatus::kOutOfRangeOutput:
      return "EQ output sample is outside normalized FLOAT32LE range";
  }
  throw std::logic_error("unhandled EQ backend process status");
}

}  // namespace fa_eq::backends

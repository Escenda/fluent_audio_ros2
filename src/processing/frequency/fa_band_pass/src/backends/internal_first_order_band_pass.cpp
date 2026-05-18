#include "fa_band_pass/backends/internal_first_order_band_pass.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fa_band_pass::backends
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

InternalFirstOrderBandPassBackend::InternalFirstOrderBandPassBackend(
  const InternalFirstOrderBandPassConfig & config)
: config_(config)
{
  if (config_.sample_rate <= 0) {
    throw std::runtime_error("feature.sample_rate must be > 0");
  }
  if (config_.channels <= 0) {
    throw std::runtime_error("feature.channels must be > 0");
  }
  const double nyquist_hz = static_cast<double>(config_.sample_rate) / 2.0;
  if (!std::isfinite(config_.low_cut_hz) || config_.low_cut_hz <= 0.0) {
    throw std::runtime_error("filter.low_cut_hz must be finite and > 0.0");
  }
  if (!std::isfinite(config_.high_cut_hz) ||
      config_.high_cut_hz <= config_.low_cut_hz ||
      config_.high_cut_hz >= nyquist_hz)
  {
    throw std::runtime_error(
      "filter.high_cut_hz must be finite, > filter.low_cut_hz, and < sample_rate / 2.0");
  }

  const double dt = 1.0 / static_cast<double>(config_.sample_rate);
  const double rc_hp = 1.0 / (2.0 * kPi * config_.low_cut_hz);
  const double rc_lp = 1.0 / (2.0 * kPi * config_.high_cut_hz);
  hp_alpha_ = rc_hp / (rc_hp + dt);
  lp_alpha_ = dt / (rc_lp + dt);
  if (!std::isfinite(hp_alpha_) || hp_alpha_ <= 0.0 || hp_alpha_ >= 1.0) {
    throw std::runtime_error("high-pass coefficient alpha must be finite and in (0.0, 1.0)");
  }
  if (!std::isfinite(lp_alpha_) || lp_alpha_ <= 0.0 || lp_alpha_ >= 1.0) {
    throw std::runtime_error("low-pass coefficient alpha must be finite and in (0.0, 1.0)");
  }

  channel_states_.assign(static_cast<size_t>(config_.channels), ChannelFilterState{});
}

double InternalFirstOrderBandPassBackend::highPassAlpha() const
{
  return hp_alpha_;
}

double InternalFirstOrderBandPassBackend::lowPassAlpha() const
{
  return lp_alpha_;
}

int InternalFirstOrderBandPassBackend::channels() const
{
  return config_.channels;
}

double InternalFirstOrderBandPassBackend::lowCutHz() const
{
  return config_.low_cut_hz;
}

double InternalFirstOrderBandPassBackend::highCutHz() const
{
  return config_.high_cut_hz;
}

ProcessStatus InternalFirstOrderBandPassBackend::process(
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
    float hp_sample = 0.0F;
    float out_sample = 0.0F;
    if (state.initialized) {
      const double high_passed =
        hp_alpha_ * (static_cast<double>(state.previous_hp_output) +
        static_cast<double>(sample) - static_cast<double>(state.previous_hp_input));
      if (!std::isfinite(high_passed) || high_passed > float_max || high_passed < -float_max) {
        return ProcessStatus::kNonFiniteOutput;
      }
      hp_sample = static_cast<float>(high_passed);

      const double low_passed =
        static_cast<double>(state.previous_lp_output) +
        lp_alpha_ * (static_cast<double>(hp_sample) - static_cast<double>(state.previous_lp_output));
      if (!std::isfinite(low_passed)) {
        return ProcessStatus::kNonFiniteOutput;
      }
      if (low_passed < kNormalizedMin || low_passed > kNormalizedMax) {
        return ProcessStatus::kOutOfRangeOutput;
      }
      out_sample = static_cast<float>(low_passed);
    }

    state.previous_hp_input = sample;
    state.previous_hp_output = hp_sample;
    state.previous_lp_output = out_sample;
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
      return "band-pass output sample is not representable as finite FLOAT32LE";
    case ProcessStatus::kOutOfRangeOutput:
      return "band-pass output sample is outside normalized FLOAT32LE range";
  }
  return "unknown band-pass backend status";
}

}  // namespace fa_band_pass::backends

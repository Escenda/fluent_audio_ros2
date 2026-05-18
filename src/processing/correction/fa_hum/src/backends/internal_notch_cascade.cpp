#include "fa_hum/backends/internal_notch_cascade.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__) || \
  (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "fa_hum internal_notch_cascade requires a little-endian target for FLOAT32LE"
#endif

namespace fa_hum::backends
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kNormalizedMin = -1.0;
constexpr double kNormalizedMax = 1.0;

bool isFinite(const double value)
{
  return std::isfinite(value);
}

bool isNormalized(const double value)
{
  return value >= kNormalizedMin && value <= kNormalizedMax;
}
}  // namespace

InternalNotchCascadeBackend::InternalNotchCascadeBackend(
  const InternalNotchCascadeConfig & config)
: config_(config)
{
  if (config_.sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  const double nyquist_hz = static_cast<double>(config_.sample_rate) / 2.0;
  if (!isFinite(config_.frequency_hz) || config_.frequency_hz <= 0.0) {
    throw std::runtime_error("hum.frequency_hz must be finite and > 0.0");
  }
  if (config_.frequency_hz >= nyquist_hz) {
    throw std::runtime_error("hum.frequency_hz must produce at least one harmonic below Nyquist");
  }
  if (config_.harmonics < 1) {
    throw std::runtime_error("hum.harmonics must be >= 1");
  }
  if (!isFinite(config_.q) || config_.q <= 0.0) {
    throw std::runtime_error("hum.q must be finite and > 0.0");
  }

  for (int harmonic = 1; harmonic <= config_.harmonics; ++harmonic) {
    const double center_hz = config_.frequency_hz * static_cast<double>(harmonic);
    if (center_hz >= nyquist_hz) {
      break;
    }

    const double omega = 2.0 * kPi * center_hz / static_cast<double>(config_.sample_rate);
    const double alpha = std::sin(omega) / (2.0 * config_.q);
    const double cos_omega = std::cos(omega);
    const double a0 = 1.0 + alpha;
    if (!isFinite(center_hz) || !isFinite(a0) || a0 == 0.0) {
      throw std::runtime_error("hum notch coefficient normalization failed because a0 is invalid");
    }

    BiquadCoefficients coefficients;
    coefficients.center_hz = center_hz;
    coefficients.b0 = 1.0 / a0;
    coefficients.b1 = (-2.0 * cos_omega) / a0;
    coefficients.b2 = 1.0 / a0;
    coefficients.a1 = (-2.0 * cos_omega) / a0;
    coefficients.a2 = (1.0 - alpha) / a0;

    if (!isFinite(coefficients.b0) || !isFinite(coefficients.b1) ||
        !isFinite(coefficients.b2) || !isFinite(coefficients.a1) ||
        !isFinite(coefficients.a2))
    {
      throw std::runtime_error("hum notch coefficient normalization produced non-finite coefficients");
    }
    cascade_coefficients_.push_back(coefficients);
  }

  if (cascade_coefficients_.empty()) {
    throw std::runtime_error("hum configuration produced no notch stages below Nyquist");
  }
  channel_states_ = zeroedChannelStates();
}

ProcessResult InternalNotchCascadeBackend::process(
  const std::string & source_id,
  const uint32_t epoch,
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output)
{
  if (source_id.empty()) {
    return ProcessResult{ProcessStatus::kEmptySourceId, false};
  }

  const size_t channel_count = static_cast<size_t>(config_.channels);
  const size_t bytes_per_frame = channel_count * sizeof(float);
  if (input.empty()) {
    return ProcessResult{ProcessStatus::kEmptyInput, false};
  }
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessResult{ProcessStatus::kMisalignedInput, false};
  }

  const bool source_changed = has_active_stream_ && active_source_id_ != source_id;
  if (has_active_stream_ && !source_changed && epoch < active_epoch_) {
    return ProcessResult{ProcessStatus::kStaleEpoch, false, false};
  }
  const bool epoch_changed = has_active_stream_ && !source_changed && epoch > active_epoch_;
  const bool source_reset = source_changed;
  const bool epoch_reset = epoch_changed;
  std::vector<ChannelCascadeState> next_states =
    (source_changed || epoch_changed) ? zeroedChannelStates() : channel_states_;
  std::vector<uint8_t> next_output(input.size());

  const size_t sample_count = input.size() / sizeof(float);
  for (size_t i = 0; i < sample_count; ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, input.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteInput, false};
    }
    if (!isNormalized(static_cast<double>(sample))) {
      return ProcessResult{ProcessStatus::kOutOfRangeInput, false};
    }

    double filtered = static_cast<double>(sample);
    ChannelCascadeState & channel_state = next_states.at(i % channel_count);
    for (size_t stage = 0; stage < cascade_coefficients_.size(); ++stage) {
      const BiquadCoefficients & coefficients = cascade_coefficients_.at(stage);
      BiquadState & state = channel_state.at(stage);
      const double stage_output =
        coefficients.b0 * filtered +
        coefficients.b1 * state.previous_input_1 +
        coefficients.b2 * state.previous_input_2 -
        coefficients.a1 * state.previous_output_1 -
        coefficients.a2 * state.previous_output_2;
      if (!isFinite(stage_output)) {
        return ProcessResult{ProcessStatus::kNonFiniteOutput, false};
      }

      state.previous_input_2 = state.previous_input_1;
      state.previous_input_1 = filtered;
      state.previous_output_2 = state.previous_output_1;
      state.previous_output_1 = stage_output;
      filtered = stage_output;
    }

    if (!isNormalized(filtered)) {
      return ProcessResult{ProcessStatus::kOutOfRangeOutput, false};
    }

    const float out_sample = static_cast<float>(filtered);
    if (!std::isfinite(out_sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput, false};
    }
    if (!isNormalized(static_cast<double>(out_sample))) {
      return ProcessResult{ProcessStatus::kOutOfRangeOutput, false};
    }
    std::memcpy(next_output.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  has_active_stream_ = true;
  active_source_id_ = source_id;
  active_epoch_ = epoch;
  channel_states_ = std::move(next_states);
  output = std::move(next_output);
  return ProcessResult{ProcessStatus::kOk, source_reset, epoch_reset};
}

size_t InternalNotchCascadeBackend::stageCount() const
{
  return cascade_coefficients_.size();
}

std::vector<double> InternalNotchCascadeBackend::centerFrequenciesHz() const
{
  std::vector<double> centers;
  centers.reserve(cascade_coefficients_.size());
  for (const BiquadCoefficients & coefficients : cascade_coefficients_) {
    centers.push_back(coefficients.center_hz);
  }
  return centers;
}

const std::string & InternalNotchCascadeBackend::activeSourceId() const
{
  return active_source_id_;
}

bool InternalNotchCascadeBackend::hasActiveStream() const
{
  return has_active_stream_;
}

uint32_t InternalNotchCascadeBackend::activeEpoch() const
{
  return active_epoch_;
}

std::vector<InternalNotchCascadeBackend::ChannelCascadeState>
InternalNotchCascadeBackend::zeroedChannelStates() const
{
  return std::vector<ChannelCascadeState>(
    static_cast<size_t>(config_.channels),
    ChannelCascadeState(cascade_coefficients_.size(), BiquadState{}));
}

const char * processStatusMessage(const ProcessStatus status)
{
  switch (status) {
    case ProcessStatus::kOk:
      return "ok";
    case ProcessStatus::kEmptySourceId:
      return "source_id is empty";
    case ProcessStatus::kEmptyInput:
      return "input data is empty";
    case ProcessStatus::kMisalignedInput:
      return "input byte length is not aligned to FLOAT32LE interleaved frames";
    case ProcessStatus::kNonFiniteInput:
      return "input sample is not finite";
    case ProcessStatus::kOutOfRangeInput:
      return "input sample is outside normalized FLOAT32LE range [-1, 1]";
    case ProcessStatus::kStaleEpoch:
      return "input epoch is older than active stream epoch";
    case ProcessStatus::kNonFiniteOutput:
      return "output sample is not finite FLOAT32LE";
    case ProcessStatus::kOutOfRangeOutput:
      return "output sample is outside normalized FLOAT32LE range [-1, 1]";
  }
  throw std::logic_error("unhandled hum backend process status");
}

}  // namespace fa_hum::backends

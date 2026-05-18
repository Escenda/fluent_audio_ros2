#include "fa_deesser/backends/internal_split_band_deesser.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fa_deesser::backends
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kMinimumAttenuationDb = -120.0;
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

double dbToLinear(double attenuation_db)
{
  return std::pow(10.0, attenuation_db / 20.0);
}
}  // namespace

InternalSplitBandDeesserBackend::InternalSplitBandDeesserBackend(
  const InternalSplitBandDeesserConfig & config)
: config_(config)
{
  if (config_.sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  const double nyquist_hz = static_cast<double>(config_.sample_rate) / 2.0;
  if (!isFinite(config_.cutoff_hz) ||
      config_.cutoff_hz <= 0.0 ||
      config_.cutoff_hz >= nyquist_hz)
  {
    throw std::runtime_error("detector.cutoff_hz must be finite and in (0.0, Nyquist)");
  }
  if (!isFinite(config_.threshold) || config_.threshold < 0.0 || config_.threshold > 1.0) {
    throw std::runtime_error("detector.threshold must be finite and in [0.0, 1.0]");
  }
  if (!isFinite(config_.attenuation_db) ||
      config_.attenuation_db < kMinimumAttenuationDb ||
      config_.attenuation_db > 0.0)
  {
    throw std::runtime_error("detector.attenuation_db must be finite and in [-120.0, 0.0]");
  }

  alpha_ =
    1.0 - std::exp((-2.0 * kPi * config_.cutoff_hz) / static_cast<double>(config_.sample_rate));
  if (!isFinite(alpha_) || alpha_ <= 0.0 || alpha_ >= 1.0) {
    throw std::runtime_error("de-esser split coefficient alpha must be finite and in (0.0, 1.0)");
  }

  attenuation_gain_ = dbToLinear(config_.attenuation_db);
  if (!isFinite(attenuation_gain_) || attenuation_gain_ < 0.0 || attenuation_gain_ > 1.0) {
    throw std::runtime_error("detector.attenuation_db produced an invalid linear gain");
  }

  low_band_state_.assign(static_cast<size_t>(config_.channels), 0.0);
}

double InternalSplitBandDeesserBackend::alpha() const
{
  return alpha_;
}

double InternalSplitBandDeesserBackend::attenuationGain() const
{
  return attenuation_gain_;
}

int InternalSplitBandDeesserBackend::channels() const
{
  return config_.channels;
}

ProcessResult InternalSplitBandDeesserBackend::process(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output,
  bool reset_state)
{
  const size_t bytes_per_frame = static_cast<size_t>(config_.channels) * sizeof(float);
  if (input.empty()) {
    return ProcessResult{ProcessStatus::kEmptyInput, 0};
  }
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessResult{ProcessStatus::kMisalignedInput, 0};
  }

  std::vector<double> next_low_band_state =
    reset_state ?
    std::vector<double>(static_cast<size_t>(config_.channels), 0.0) :
    low_band_state_;
  std::vector<uint8_t> next_output(input.size());
  const double float_max = static_cast<double>(std::numeric_limits<float>::max());
  uint64_t attenuated_in_frame = 0;

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

    double & channel_low_state =
      next_low_band_state.at(i % static_cast<size_t>(config_.channels));
    const double input_sample = static_cast<double>(sample);
    const double low_band = channel_low_state + (alpha_ * (input_sample - channel_low_state));
    if (!isFinite(low_band) || low_band > float_max || low_band < -float_max) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput, 0};
    }
    const double high_band = input_sample - low_band;
    if (!isFinite(high_band) || high_band > float_max || high_band < -float_max) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput, 0};
    }
    channel_low_state = low_band;

    double processed_high_band = high_band;
    if (std::abs(high_band) >= config_.threshold) {
      processed_high_band = high_band * attenuation_gain_;
      ++attenuated_in_frame;
    }

    const double output_sample = low_band + processed_high_band;
    if (!isFinite(output_sample) || output_sample > float_max || output_sample < -float_max) {
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

  low_band_state_ = std::move(next_low_band_state);
  output = std::move(next_output);
  return ProcessResult{ProcessStatus::kOk, attenuated_in_frame};
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
      return "de-esser output sample is not representable as finite FLOAT32LE";
    case ProcessStatus::kOutOfRangeOutput:
      return "de-esser output sample is outside normalized FLOAT32LE range";
  }
  return "unknown de-esser backend status";
}

}  // namespace fa_deesser::backends

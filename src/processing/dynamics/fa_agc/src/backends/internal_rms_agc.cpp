#include "fa_agc/backends/internal_rms_agc.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_agc::backends
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

InternalRmsAgcConfig::InternalRmsAgcConfig(
  int sample_rate_value,
  int channels_value,
  double target_rms_value,
  double min_gain_value,
  double max_gain_value,
  double attack_ms_value,
  double release_ms_value)
: sample_rate(sample_rate_value),
  channels(channels_value),
  target_rms(target_rms_value),
  min_gain(min_gain_value),
  max_gain(max_gain_value),
  attack_ms(attack_ms_value),
  release_ms(release_ms_value)
{
}

InternalRmsAgcBackend::InternalRmsAgcBackend(const InternalRmsAgcConfig & config)
: config_(config)
{
  if (config_.sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (!isFinite(config_.target_rms) ||
      config_.target_rms <= 0.0 ||
      config_.target_rms > 1.0)
  {
    throw std::runtime_error("agc.target_rms must be finite and in (0.0, 1.0]");
  }
  if (!isFinite(config_.min_gain) || config_.min_gain <= 0.0) {
    throw std::runtime_error("agc.min_gain must be finite and > 0.0");
  }
  if (!isFinite(config_.max_gain) || config_.max_gain < config_.min_gain) {
    throw std::runtime_error("agc.max_gain must be finite and >= agc.min_gain");
  }
  if (config_.min_gain > 1.0 || config_.max_gain < 1.0) {
    throw std::runtime_error("agc.min_gain <= 1.0 <= agc.max_gain is required for initial gain");
  }
  if (!isFinite(config_.attack_ms) || config_.attack_ms <= 0.0) {
    throw std::runtime_error("agc.attack_ms must be finite and > 0.0");
  }
  if (!isFinite(config_.release_ms) || config_.release_ms <= 0.0) {
    throw std::runtime_error("agc.release_ms must be finite and > 0.0");
  }
}

double InternalRmsAgcBackend::targetRms() const
{
  return config_.target_rms;
}

double InternalRmsAgcBackend::minGain() const
{
  return config_.min_gain;
}

double InternalRmsAgcBackend::maxGain() const
{
  return config_.max_gain;
}

double InternalRmsAgcBackend::attackMs() const
{
  return config_.attack_ms;
}

double InternalRmsAgcBackend::releaseMs() const
{
  return config_.release_ms;
}

double InternalRmsAgcBackend::currentGain() const
{
  return current_gain_;
}

double InternalRmsAgcBackend::lastFrameRms() const
{
  return last_frame_rms_;
}

double InternalRmsAgcBackend::lastTargetGain() const
{
  return last_target_gain_;
}

double InternalRmsAgcBackend::calculateFrameRms(const std::vector<float> & samples) const
{
  double square_sum = 0.0;
  for (const float sample : samples) {
    const double value = static_cast<double>(sample);
    square_sum += value * value;
  }

  const double mean_square = square_sum / static_cast<double>(samples.size());
  return std::sqrt(mean_square);
}

double InternalRmsAgcBackend::boundedTargetGain(double frame_rms) const
{
  double target_gain = config_.max_gain;
  if (frame_rms > 0.0) {
    target_gain = config_.target_rms / frame_rms;
  }

  if (target_gain < config_.min_gain) {
    return config_.min_gain;
  }
  if (target_gain > config_.max_gain) {
    return config_.max_gain;
  }
  return target_gain;
}

double InternalRmsAgcBackend::smoothingAlpha(double time_constant_ms, size_t sample_count) const
{
  const double frame_count =
    static_cast<double>(sample_count) / static_cast<double>(config_.channels);
  const double frame_seconds = frame_count / static_cast<double>(config_.sample_rate);
  const double time_constant_seconds = time_constant_ms / 1000.0;
  return 1.0 - std::exp(-frame_seconds / time_constant_seconds);
}

double InternalRmsAgcBackend::smoothedGain(double target_gain, size_t sample_count) const
{
  const double time_constant_ms = target_gain < current_gain_ ?
    config_.attack_ms :
    config_.release_ms;
  const double alpha = smoothingAlpha(time_constant_ms, sample_count);
  if (!isFinite(alpha)) {
    return alpha;
  }

  const double next_gain = current_gain_ + (alpha * (target_gain - current_gain_));
  if (next_gain < config_.min_gain) {
    return config_.min_gain;
  }
  if (next_gain > config_.max_gain) {
    return config_.max_gain;
  }
  return next_gain;
}

ProcessResult InternalRmsAgcBackend::process(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output)
{
  const size_t bytes_per_frame = static_cast<size_t>(config_.channels) * sizeof(float);
  if (input.empty()) {
    return ProcessResult{
      ProcessStatus::kEmptyInput, last_frame_rms_, last_target_gain_, current_gain_,
      GainDirection::kUnchanged};
  }
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessResult{
      ProcessStatus::kMisalignedInput, last_frame_rms_, last_target_gain_, current_gain_,
      GainDirection::kUnchanged};
  }

  std::vector<float> samples(input.size() / sizeof(float));
  for (size_t i = 0; i < samples.size(); ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, input.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample)) {
      return ProcessResult{
        ProcessStatus::kNonFiniteInput, last_frame_rms_, last_target_gain_, current_gain_,
        GainDirection::kUnchanged};
    }
    if (!isNormalizedSample(sample)) {
      return ProcessResult{
        ProcessStatus::kOutOfRangeInput, last_frame_rms_, last_target_gain_, current_gain_,
        GainDirection::kUnchanged};
    }
    samples[i] = sample;
  }

  const double frame_rms = calculateFrameRms(samples);
  if (!isFinite(frame_rms)) {
    return ProcessResult{
      ProcessStatus::kNonFiniteOutput, last_frame_rms_, last_target_gain_, current_gain_,
      GainDirection::kUnchanged};
  }

  const double target_gain = boundedTargetGain(frame_rms);
  const double candidate_gain = smoothedGain(target_gain, samples.size());
  if (!isFinite(target_gain) || !isFinite(candidate_gain)) {
    return ProcessResult{
      ProcessStatus::kNonFiniteGain, last_frame_rms_, last_target_gain_, current_gain_,
      GainDirection::kUnchanged};
  }

  std::vector<uint8_t> next_output(input.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    const double output_sample = static_cast<double>(samples[i]) * candidate_gain;
    if (!isFinite(output_sample)) {
      return ProcessResult{
        ProcessStatus::kNonFiniteOutput, last_frame_rms_, last_target_gain_, current_gain_,
        GainDirection::kUnchanged};
    }
    if (output_sample < kNormalizedMin || output_sample > kNormalizedMax) {
      return ProcessResult{
        ProcessStatus::kOutOfRangeOutput, last_frame_rms_, last_target_gain_, current_gain_,
        GainDirection::kUnchanged};
    }

    const float out_sample = static_cast<float>(output_sample);
    if (!isNormalizedSample(out_sample)) {
      return ProcessResult{
        ProcessStatus::kOutOfRangeOutput, last_frame_rms_, last_target_gain_, current_gain_,
        GainDirection::kUnchanged};
    }
    std::memcpy(next_output.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  GainDirection direction = GainDirection::kUnchanged;
  if (candidate_gain < current_gain_) {
    direction = GainDirection::kReduction;
  } else if (candidate_gain > current_gain_) {
    direction = GainDirection::kIncrease;
  }

  current_gain_ = candidate_gain;
  last_frame_rms_ = frame_rms;
  last_target_gain_ = target_gain;
  output = std::move(next_output);
  return ProcessResult{ProcessStatus::kOk, frame_rms, target_gain, current_gain_, direction};
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
      return "AGC gain is not finite";
    case ProcessStatus::kNonFiniteOutput:
      return "AGC output sample is not representable as finite FLOAT32LE";
    case ProcessStatus::kOutOfRangeOutput:
      return "AGC output sample is outside normalized FLOAT32LE range";
  }
  return "unknown AGC backend status";
}

}  // namespace fa_agc::backends

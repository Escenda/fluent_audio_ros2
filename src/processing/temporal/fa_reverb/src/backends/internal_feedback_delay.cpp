#include "fa_reverb/backends/internal_feedback_delay.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fa_reverb::backends
{

namespace
{
constexpr float kSilenceSample = 0.0F;
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;
constexpr int kMaxExpectedSampleRate = 384000;
constexpr int kMaxExpectedChannels = 64;
constexpr double kMinFeedbackGain = 0.20;
constexpr double kFeedbackGainRange = 0.65;
constexpr std::array<double, 4> kBaseDelayMs = {29.7, 37.1, 41.1, 43.7};

bool isFinite(double value)
{
  return std::isfinite(value);
}

bool isNormalizedSample(float value)
{
  return std::isfinite(value) && value >= kMinNormalizedSample && value <= kMaxNormalizedSample;
}

float readFloat32Le(const std::vector<uint8_t> & data, size_t sample_index)
{
  const size_t offset = sample_index * sizeof(float);
  const uint32_t bits =
    static_cast<uint32_t>(data[offset]) |
    (static_cast<uint32_t>(data[offset + 1U]) << 8U) |
    (static_cast<uint32_t>(data[offset + 2U]) << 16U) |
    (static_cast<uint32_t>(data[offset + 3U]) << 24U);

  float sample = 0.0F;
  std::memcpy(&sample, &bits, sizeof(sample));
  return sample;
}

void writeFloat32Le(std::vector<uint8_t> & data, size_t sample_index, float sample)
{
  uint32_t bits = 0U;
  std::memcpy(&bits, &sample, sizeof(bits));
  const size_t offset = sample_index * sizeof(float);
  data[offset] = static_cast<uint8_t>(bits & 0xFFU);
  data[offset + 1U] = static_cast<uint8_t>((bits >> 8U) & 0xFFU);
  data[offset + 2U] = static_cast<uint8_t>((bits >> 16U) & 0xFFU);
  data[offset + 3U] = static_cast<uint8_t>((bits >> 24U) & 0xFFU);
}
}  // namespace

InternalFeedbackDelayBackend::InternalFeedbackDelayBackend(
  const InternalFeedbackDelayConfig & config)
: config_(config)
{
  if (config_.sample_rate <= 0 || config_.sample_rate > kMaxExpectedSampleRate) {
    throw std::runtime_error("expected.sample_rate must satisfy 0 < value <= 384000");
  }
  if (config_.channels <= 0 || config_.channels > kMaxExpectedChannels) {
    throw std::runtime_error("expected.channels must satisfy 0 < value <= 64");
  }
  if (!isFinite(config_.room_size) || config_.room_size < 0.0 || config_.room_size > 1.0) {
    throw std::runtime_error("reverb.room_size must be finite and satisfy 0.0 <= value <= 1.0");
  }
  if (!isFinite(config_.damping) || config_.damping < 0.0 || config_.damping > 1.0) {
    throw std::runtime_error("reverb.damping must be finite and satisfy 0.0 <= value <= 1.0");
  }
  if (!isFinite(config_.wet_gain) || config_.wet_gain < 0.0 || config_.wet_gain > 1.0) {
    throw std::runtime_error("reverb.wet_gain must be finite and satisfy 0.0 <= value <= 1.0");
  }
  if (!isFinite(config_.dry_gain) || config_.dry_gain < 0.0 || config_.dry_gain > 1.0) {
    throw std::runtime_error("reverb.dry_gain must be finite and satisfy 0.0 <= value <= 1.0");
  }
  if ((config_.wet_gain + config_.dry_gain) > 1.0) {
    throw std::runtime_error("reverb.wet_gain + reverb.dry_gain must be <= 1.0");
  }

  effective_feedback_gain_ = kMinFeedbackGain + (kFeedbackGainRange * config_.room_size);
  delay_samples_.reserve(kBaseDelayMs.size());
  for (const double delay_ms : kBaseDelayMs) {
    const double raw_delay_samples =
      delay_ms * static_cast<double>(config_.sample_rate) / 1000.0;
    if (!isFinite(raw_delay_samples) ||
        raw_delay_samples > static_cast<double>(std::numeric_limits<long long>::max()))
    {
      throw std::runtime_error("reverb delay line converts to an unsupported sample count");
    }
    const size_t delay_samples = static_cast<size_t>(std::llround(raw_delay_samples));
    if (delay_samples == 0U) {
      throw std::runtime_error("reverb delay line must convert to at least 1 sample");
    }
    delay_samples_.push_back(delay_samples);
  }

  resetReverbState(delay_lines_);
}

int InternalFeedbackDelayBackend::sampleRate() const
{
  return config_.sample_rate;
}

int InternalFeedbackDelayBackend::channels() const
{
  return config_.channels;
}

double InternalFeedbackDelayBackend::roomSize() const
{
  return config_.room_size;
}

double InternalFeedbackDelayBackend::damping() const
{
  return config_.damping;
}

double InternalFeedbackDelayBackend::wetGain() const
{
  return config_.wet_gain;
}

double InternalFeedbackDelayBackend::dryGain() const
{
  return config_.dry_gain;
}

double InternalFeedbackDelayBackend::effectiveFeedbackGain() const
{
  return effective_feedback_gain_;
}

size_t InternalFeedbackDelayBackend::delayLineCount() const
{
  return delay_samples_.size();
}

const std::string & InternalFeedbackDelayBackend::currentSourceId() const
{
  return current_source_id_;
}

void InternalFeedbackDelayBackend::resetReverbState(
  std::vector<std::vector<DelayLineState>> & state) const
{
  state.assign(
    static_cast<size_t>(config_.channels),
    std::vector<DelayLineState>(delay_samples_.size()));

  for (auto & channel_state : state) {
    for (size_t line_index = 0; line_index < delay_samples_.size(); ++line_index) {
      channel_state[line_index].buffer.assign(delay_samples_[line_index], kSilenceSample);
      channel_state[line_index].position = 0U;
      channel_state[line_index].filter_state = kSilenceSample;
    }
  }
}

bool InternalFeedbackDelayBackend::validateReverbState(
  const std::vector<std::vector<DelayLineState>> & state) const
{
  if (state.size() != static_cast<size_t>(config_.channels)) {
    return false;
  }
  for (const auto & channel_state : state) {
    if (channel_state.size() != delay_samples_.size()) {
      return false;
    }
    for (size_t line_index = 0; line_index < channel_state.size(); ++line_index) {
      const DelayLineState & line = channel_state[line_index];
      if (line.buffer.size() != delay_samples_[line_index] ||
          line.position >= line.buffer.size() ||
          !isNormalizedSample(line.filter_state))
      {
        return false;
      }
    }
  }
  return true;
}

ProcessResult InternalFeedbackDelayBackend::process(
  const std::string & source_id,
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output)
{
  if (source_id.empty()) {
    return ProcessResult{ProcessStatus::kEmptySourceId, false};
  }
  if (input.empty()) {
    return ProcessResult{ProcessStatus::kEmptyInput, false};
  }

  const size_t channels = static_cast<size_t>(config_.channels);
  const size_t bytes_per_frame = channels * sizeof(float);
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessResult{ProcessStatus::kMisalignedInput, false};
  }

  std::vector<std::vector<DelayLineState>> next_state = delay_lines_;
  const bool needs_initialization = current_source_id_.empty();
  const bool source_changed = !current_source_id_.empty() && source_id != current_source_id_;
  if (needs_initialization || source_changed) {
    resetReverbState(next_state);
  }
  if (!validateReverbState(next_state)) {
    return ProcessResult{ProcessStatus::kInvalidState, false};
  }

  std::vector<uint8_t> next_output(input.size());
  const double float_max = static_cast<double>(std::numeric_limits<float>::max());
  const size_t frame_count = input.size() / bytes_per_frame;
  const double input_gain = 1.0 - effective_feedback_gain_;
  for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
    for (size_t channel_index = 0; channel_index < channels; ++channel_index) {
      const size_t sample_index = (frame_index * channels) + channel_index;
      const float input_sample = readFloat32Le(input, sample_index);
      if (!std::isfinite(input_sample)) {
        return ProcessResult{ProcessStatus::kNonFiniteInput, false};
      }
      if (!isNormalizedSample(input_sample)) {
        return ProcessResult{ProcessStatus::kOutOfRangeInput, false};
      }

      double wet_sum = 0.0;
      for (DelayLineState & line : next_state[channel_index]) {
        const size_t delay_index = line.position;
        const float delayed_sample = line.buffer[delay_index];
        wet_sum += static_cast<double>(delayed_sample);

        const double filtered_sample =
          ((1.0 - config_.damping) * static_cast<double>(delayed_sample)) +
          (config_.damping * static_cast<double>(line.filter_state));
        const double next_feedback_state =
          (input_gain * static_cast<double>(input_sample)) +
          (effective_feedback_gain_ * filtered_sample);

        if (!isFinite(filtered_sample) || filtered_sample > float_max ||
            filtered_sample < -float_max || !isFinite(next_feedback_state) ||
            next_feedback_state > float_max || next_feedback_state < -float_max)
        {
          return ProcessResult{ProcessStatus::kNonFiniteOutput, false};
        }

        const float filtered_float = static_cast<float>(filtered_sample);
        const float next_state_float = static_cast<float>(next_feedback_state);
        if (!isNormalizedSample(filtered_float) || !isNormalizedSample(next_state_float)) {
          return ProcessResult{ProcessStatus::kOutOfRangeOutput, false};
        }

        line.buffer[delay_index] = next_state_float;
        line.filter_state = filtered_float;
        line.position = (delay_index + 1U) % line.buffer.size();
      }

      const double wet_sample = wet_sum / static_cast<double>(delay_samples_.size());
      const double output_sample =
        (config_.dry_gain * static_cast<double>(input_sample)) +
        (config_.wet_gain * wet_sample);

      if (!isFinite(output_sample) || output_sample > float_max || output_sample < -float_max) {
        return ProcessResult{ProcessStatus::kNonFiniteOutput, false};
      }

      const float output_float = static_cast<float>(output_sample);
      if (!isNormalizedSample(output_float)) {
        return ProcessResult{ProcessStatus::kOutOfRangeOutput, false};
      }
      writeFloat32Le(next_output, sample_index, output_float);
    }
  }

  output = std::move(next_output);
  delay_lines_ = std::move(next_state);
  current_source_id_ = source_id;
  return ProcessResult{ProcessStatus::kOk, source_changed};
}

const char * processStatusMessage(ProcessStatus status)
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
    case ProcessStatus::kInvalidState:
      return "reverb delay state does not match configured channels or delay lines";
    case ProcessStatus::kNonFiniteInput:
      return "input sample is not finite";
    case ProcessStatus::kOutOfRangeInput:
      return "input sample is outside normalized FLOAT32LE range";
    case ProcessStatus::kNonFiniteOutput:
      return "reverb output or state is not representable as finite FLOAT32LE";
    case ProcessStatus::kOutOfRangeOutput:
      return "reverb output or state is outside normalized FLOAT32LE range";
  }
  throw std::logic_error("unhandled reverb backend process status");
}

}  // namespace fa_reverb::backends

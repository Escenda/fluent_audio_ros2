#include "fa_echo/backends/internal_feedback_echo.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fa_echo::backends
{

namespace
{
constexpr float kSilenceSample = 0.0F;
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;

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

InternalFeedbackEchoBackend::InternalFeedbackEchoBackend(
  const InternalFeedbackEchoConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.delay_samples == 0U) {
    throw std::runtime_error("echo.delay_ms must convert to at least 1 sample");
  }
  if (!isFinite(config_.feedback_gain) || std::abs(config_.feedback_gain) >= 1.0) {
    throw std::runtime_error("echo.feedback_gain must be finite and satisfy abs(value) < 1.0");
  }
  if (!isFinite(config_.wet_gain)) {
    throw std::runtime_error("echo.wet_gain must be finite");
  }
  if (!isFinite(config_.dry_gain)) {
    throw std::runtime_error("echo.dry_gain must be finite");
  }

  resetDelayState(delay_buffers_, delay_positions_);
}

int InternalFeedbackEchoBackend::channels() const
{
  return config_.channels;
}

size_t InternalFeedbackEchoBackend::delaySamples() const
{
  return config_.delay_samples;
}

double InternalFeedbackEchoBackend::feedbackGain() const
{
  return config_.feedback_gain;
}

double InternalFeedbackEchoBackend::wetGain() const
{
  return config_.wet_gain;
}

double InternalFeedbackEchoBackend::dryGain() const
{
  return config_.dry_gain;
}

const std::string & InternalFeedbackEchoBackend::currentSourceId() const
{
  return current_source_id_;
}

void InternalFeedbackEchoBackend::resetDelayState(
  std::vector<std::vector<float>> & buffers,
  std::vector<size_t> & positions) const
{
  buffers.assign(
    static_cast<size_t>(config_.channels),
    std::vector<float>(config_.delay_samples, kSilenceSample));
  positions.assign(static_cast<size_t>(config_.channels), 0U);
}

ProcessResult InternalFeedbackEchoBackend::process(
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

  std::vector<std::vector<float>> next_buffers = delay_buffers_;
  std::vector<size_t> next_positions = delay_positions_;
  const bool needs_initialization = current_source_id_.empty();
  const bool source_changed = !current_source_id_.empty() && source_id != current_source_id_;
  if (needs_initialization || source_changed) {
    resetDelayState(next_buffers, next_positions);
  }

  if (next_buffers.size() != channels || next_positions.size() != channels) {
    return ProcessResult{ProcessStatus::kInvalidState, false};
  }
  for (size_t channel_index = 0; channel_index < channels; ++channel_index) {
    if (next_buffers[channel_index].size() != config_.delay_samples ||
        next_positions[channel_index] >= config_.delay_samples)
    {
      return ProcessResult{ProcessStatus::kInvalidState, false};
    }
  }

  std::vector<uint8_t> next_output(input.size());
  const double float_max = static_cast<double>(std::numeric_limits<float>::max());
  const size_t frame_count = input.size() / bytes_per_frame;
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

      const size_t delay_index = next_positions[channel_index];
      const float delayed_sample = next_buffers[channel_index][delay_index];
      const double output_sample =
        (config_.dry_gain * static_cast<double>(input_sample)) +
        (config_.wet_gain * static_cast<double>(delayed_sample));
      const double next_state =
        static_cast<double>(input_sample) +
        (config_.feedback_gain * static_cast<double>(delayed_sample));

      if (!isFinite(output_sample) || output_sample > float_max || output_sample < -float_max ||
          !isFinite(next_state) || next_state > float_max || next_state < -float_max)
      {
        return ProcessResult{ProcessStatus::kNonFiniteOutput, false};
      }

      const float output_float = static_cast<float>(output_sample);
      const float next_state_float = static_cast<float>(next_state);
      if (!isNormalizedSample(output_float)) {
        return ProcessResult{ProcessStatus::kOutOfRangeOutput, false};
      }
      if (!isNormalizedSample(next_state_float)) {
        return ProcessResult{ProcessStatus::kOutOfRangeOutput, false};
      }

      writeFloat32Le(next_output, sample_index, output_float);
      next_buffers[channel_index][delay_index] = next_state_float;
      next_positions[channel_index] = (delay_index + 1U) % config_.delay_samples;
    }
  }

  output = std::move(next_output);
  delay_buffers_ = std::move(next_buffers);
  delay_positions_ = std::move(next_positions);
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
      return "delay state does not match configured channels or delay";
    case ProcessStatus::kNonFiniteInput:
      return "input sample is not finite";
    case ProcessStatus::kOutOfRangeInput:
      return "input sample is outside normalized FLOAT32LE range";
    case ProcessStatus::kNonFiniteOutput:
      return "echo output or state is not representable as finite FLOAT32LE";
    case ProcessStatus::kOutOfRangeOutput:
      return "echo output or state is outside normalized FLOAT32LE range";
  }
  throw std::logic_error("unhandled echo backend process status");
}

}  // namespace fa_echo::backends

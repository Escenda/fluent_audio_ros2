#include "fa_delay/backends/internal_sample_delay.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_delay::backends
{

namespace
{
constexpr float kSilenceSample = 0.0F;
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;

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

InternalSampleDelayBackend::InternalSampleDelayBackend(
  const InternalSampleDelayConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.delay_samples == 0U) {
    throw std::runtime_error("delay.ms must convert to at least 1 sample");
  }

  resetDelayState(delay_buffers_);
}

int InternalSampleDelayBackend::channels() const
{
  return config_.channels;
}

size_t InternalSampleDelayBackend::delaySamples() const
{
  return config_.delay_samples;
}

const std::string & InternalSampleDelayBackend::currentSourceId() const
{
  return current_source_id_;
}

void InternalSampleDelayBackend::resetDelayState(
  std::vector<std::deque<float>> & buffers) const
{
  buffers.assign(
    static_cast<size_t>(config_.channels),
    std::deque<float>(config_.delay_samples, kSilenceSample));
}

bool InternalSampleDelayBackend::validateDelayState(
  const std::vector<std::deque<float>> & buffers) const
{
  if (buffers.size() != static_cast<size_t>(config_.channels)) {
    return false;
  }
  for (const auto & channel_buffer : buffers) {
    if (channel_buffer.size() != config_.delay_samples) {
      return false;
    }
  }
  return true;
}

ProcessResult InternalSampleDelayBackend::process(
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

  std::vector<std::deque<float>> next_buffers = delay_buffers_;
  const bool needs_initialization = current_source_id_.empty();
  const bool source_changed = !current_source_id_.empty() && source_id != current_source_id_;
  if (needs_initialization || source_changed) {
    resetDelayState(next_buffers);
  }
  if (!validateDelayState(next_buffers)) {
    return ProcessResult{ProcessStatus::kInvalidState, false};
  }

  std::vector<uint8_t> next_output(input.size());
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

      const float delayed_sample = next_buffers[channel_index].front();
      next_buffers[channel_index].pop_front();
      next_buffers[channel_index].push_back(input_sample);
      writeFloat32Le(next_output, sample_index, delayed_sample);
    }
  }

  output = std::move(next_output);
  delay_buffers_ = std::move(next_buffers);
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
  }
  throw std::logic_error("unhandled delay backend process status");
}

}  // namespace fa_delay::backends

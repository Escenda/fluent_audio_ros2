#include "fa_fade/backends/internal_linear_fade.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fa_fade::backends
{

namespace
{
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

InternalLinearFadeBackend::InternalLinearFadeBackend(
  const InternalLinearFadeConfig & config)
: config_(config),
  position_frames_(config.initial_position_frames)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.duration_frames == 0U) {
    throw std::runtime_error("fade.duration_frames must be > 0");
  }
  static_cast<void>(fadeModeName(config_.mode));
}

int InternalLinearFadeBackend::channels() const
{
  return config_.channels;
}

FadeMode InternalLinearFadeBackend::mode() const
{
  return config_.mode;
}

size_t InternalLinearFadeBackend::durationFrames() const
{
  return config_.duration_frames;
}

uint64_t InternalLinearFadeBackend::positionFrames() const
{
  return position_frames_;
}

double InternalLinearFadeBackend::gainAtPosition(uint64_t position_frames) const
{
  const double progress =
    static_cast<double>(position_frames) / static_cast<double>(config_.duration_frames);
  if (config_.mode == FadeMode::kFadeIn) {
    if (progress >= 1.0) {
      return 1.0;
    }
    return progress;
  }
  if (progress >= 1.0) {
    return 0.0;
  }
  return 1.0 - progress;
}

ProcessResult InternalLinearFadeBackend::process(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output)
{
  if (input.empty()) {
    return ProcessResult{ProcessStatus::kEmptyInput, 0U, 0U, position_frames_};
  }

  const size_t channels = static_cast<size_t>(config_.channels);
  const size_t bytes_per_frame = channels * sizeof(float);
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessResult{ProcessStatus::kMisalignedInput, 0U, 0U, position_frames_};
  }

  const size_t frame_count = input.size() / bytes_per_frame;
  if (static_cast<uint64_t>(std::numeric_limits<uint64_t>::max() - position_frames_) <
      static_cast<uint64_t>(frame_count))
  {
    return ProcessResult{ProcessStatus::kPositionOverflow, frame_count, 0U, position_frames_};
  }

  std::vector<uint8_t> next_output(input.size());
  const size_t sample_count = input.size() / sizeof(float);
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const float sample = readFloat32Le(input, sample_index);
    if (!std::isfinite(sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteInput, frame_count, 0U, position_frames_};
    }
    if (!isNormalizedSample(sample)) {
      return ProcessResult{ProcessStatus::kOutOfRangeInput, frame_count, 0U, position_frames_};
    }

    const uint64_t sample_position_frames =
      position_frames_ + static_cast<uint64_t>(sample_index / channels);
    const double faded = static_cast<double>(sample) * gainAtPosition(sample_position_frames);
    if (!std::isfinite(faded)) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput, frame_count, 0U, position_frames_};
    }
    if (faded < static_cast<double>(kMinNormalizedSample) ||
        faded > static_cast<double>(kMaxNormalizedSample))
    {
      return ProcessResult{ProcessStatus::kOutOfRangeOutput, frame_count, 0U, position_frames_};
    }

    const float output_sample = static_cast<float>(faded);
    if (!std::isfinite(output_sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput, frame_count, 0U, position_frames_};
    }
    writeFloat32Le(next_output, sample_index, output_sample);
  }

  output = std::move(next_output);
  position_frames_ += static_cast<uint64_t>(frame_count);
  return ProcessResult{ProcessStatus::kOk, frame_count, frame_count, position_frames_};
}

const char * fadeModeName(FadeMode mode)
{
  switch (mode) {
    case FadeMode::kFadeIn:
      return "fade_in";
    case FadeMode::kFadeOut:
      return "fade_out";
  }
  throw std::logic_error("unhandled fade mode");
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
    case ProcessStatus::kPositionOverflow:
      return "fade position would overflow uint64";
    case ProcessStatus::kNonFiniteInput:
      return "input sample is not finite";
    case ProcessStatus::kOutOfRangeInput:
      return "input sample is outside normalized FLOAT32LE range";
    case ProcessStatus::kNonFiniteOutput:
      return "fade output sample is not finite";
    case ProcessStatus::kOutOfRangeOutput:
      return "fade output sample is outside normalized FLOAT32LE range";
  }
  throw std::logic_error("unhandled fade backend process status");
}

}  // namespace fa_fade::backends

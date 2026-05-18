#include "fa_window/backends/internal_window_function.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_window::backends
{

namespace
{
constexpr double kPi = 3.141592653589793238462643383279502884;
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

InternalWindowFunctionBackend::InternalWindowFunctionBackend(
  const InternalWindowFunctionConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_frames <= 1U) {
    throw std::runtime_error("window.expected_frames must be > 1");
  }
  static_cast<void>(windowTypeName(config_.window_type));
}

int InternalWindowFunctionBackend::channels() const
{
  return config_.channels;
}

WindowType InternalWindowFunctionBackend::windowType() const
{
  return config_.window_type;
}

size_t InternalWindowFunctionBackend::expectedFrames() const
{
  return config_.expected_frames;
}

bool InternalWindowFunctionBackend::strictFrameCount() const
{
  return config_.strict_frame_count;
}

double InternalWindowFunctionBackend::coefficientAt(
  size_t frame_index,
  size_t frame_count) const
{
  const double phase =
    (2.0 * kPi * static_cast<double>(frame_index)) / static_cast<double>(frame_count - 1U);
  switch (config_.window_type) {
    case WindowType::kHann:
      return 0.5 * (1.0 - std::cos(phase));
    case WindowType::kHamming:
      return 0.54 - (0.46 * std::cos(phase));
  }
  throw std::logic_error("unhandled window type");
}

ProcessResult InternalWindowFunctionBackend::process(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output) const
{
  if (input.empty()) {
    return ProcessResult{ProcessStatus::kEmptyInput, 0U, 0U};
  }

  const size_t channels = static_cast<size_t>(config_.channels);
  const size_t bytes_per_frame = channels * sizeof(float);
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessResult{ProcessStatus::kMisalignedInput, 0U, 0U};
  }

  const size_t frame_count = input.size() / bytes_per_frame;
  if (config_.strict_frame_count && frame_count != config_.expected_frames) {
    return ProcessResult{ProcessStatus::kFrameCountMismatch, frame_count, 0U};
  }
  if (!config_.strict_frame_count && frame_count <= 1U) {
    return ProcessResult{ProcessStatus::kTooFewFrames, frame_count, 0U};
  }

  std::vector<uint8_t> next_output(input.size());
  const size_t sample_count = input.size() / sizeof(float);
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const float sample = readFloat32Le(input, sample_index);
    if (!std::isfinite(sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteInput, frame_count, 0U};
    }
    if (!isNormalizedSample(sample)) {
      return ProcessResult{ProcessStatus::kOutOfRangeInput, frame_count, 0U};
    }

    const size_t frame_index = sample_index / channels;
    const double windowed = static_cast<double>(sample) * coefficientAt(frame_index, frame_count);
    if (!std::isfinite(windowed)) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput, frame_count, 0U};
    }
    if (windowed < static_cast<double>(kMinNormalizedSample) ||
        windowed > static_cast<double>(kMaxNormalizedSample))
    {
      return ProcessResult{ProcessStatus::kOutOfRangeOutput, frame_count, 0U};
    }

    const float output_sample = static_cast<float>(windowed);
    if (!std::isfinite(output_sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput, frame_count, 0U};
    }
    writeFloat32Le(next_output, sample_index, output_sample);
  }

  output = std::move(next_output);
  return ProcessResult{ProcessStatus::kOk, frame_count, frame_count};
}

const char * windowTypeName(WindowType window_type)
{
  switch (window_type) {
    case WindowType::kHann:
      return "hann";
    case WindowType::kHamming:
      return "hamming";
  }
  throw std::logic_error("unhandled window type");
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
    case ProcessStatus::kFrameCountMismatch:
      return "input frame count does not match configured strict window frame count";
    case ProcessStatus::kTooFewFrames:
      return "input frame count must be > 1";
    case ProcessStatus::kNonFiniteInput:
      return "input sample is not finite";
    case ProcessStatus::kOutOfRangeInput:
      return "input sample is outside normalized FLOAT32LE range";
    case ProcessStatus::kNonFiniteOutput:
      return "window output sample is not finite";
    case ProcessStatus::kOutOfRangeOutput:
      return "window output sample is outside normalized FLOAT32LE range";
  }
  throw std::logic_error("unhandled window backend process status");
}

}  // namespace fa_window::backends

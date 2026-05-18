#include "fa_trim/backends/internal_frame_trim.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_trim::backends
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
}  // namespace

InternalFrameTrimBackend::InternalFrameTrimBackend(
  const InternalFrameTrimConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.leading_frames == 0U && config_.trailing_frames == 0U) {
    throw std::runtime_error(
      "at least one of trim.leading_frames or trim.trailing_frames must be > 0");
  }
}

int InternalFrameTrimBackend::channels() const
{
  return config_.channels;
}

size_t InternalFrameTrimBackend::leadingFrames() const
{
  return config_.leading_frames;
}

size_t InternalFrameTrimBackend::trailingFrames() const
{
  return config_.trailing_frames;
}

ProcessResult InternalFrameTrimBackend::process(
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
  const size_t sample_count = input.size() / sizeof(float);
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const float sample = readFloat32Le(input, sample_index);
    if (!std::isfinite(sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteInput, frame_count, 0U};
    }
    if (!isNormalizedSample(sample)) {
      return ProcessResult{ProcessStatus::kOutOfRangeInput, frame_count, 0U};
    }
  }

  if (config_.leading_frames >= frame_count ||
      config_.trailing_frames >= (frame_count - config_.leading_frames))
  {
    return ProcessResult{ProcessStatus::kTrimExhaustsInput, frame_count, 0U};
  }

  const size_t output_frame_count =
    frame_count - config_.leading_frames - config_.trailing_frames;
  const size_t start_byte = config_.leading_frames * bytes_per_frame;
  const size_t byte_count = output_frame_count * bytes_per_frame;

  std::vector<uint8_t> next_output;
  next_output.assign(
    input.begin() + static_cast<std::ptrdiff_t>(start_byte),
    input.begin() + static_cast<std::ptrdiff_t>(start_byte + byte_count));

  output = std::move(next_output);
  return ProcessResult{ProcessStatus::kOk, frame_count, output_frame_count};
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
    case ProcessStatus::kTrimExhaustsInput:
      return "trim removes all sample frames";
    case ProcessStatus::kNonFiniteInput:
      return "input sample is not finite";
    case ProcessStatus::kOutOfRangeInput:
      return "input sample is outside normalized FLOAT32LE range";
  }
  throw std::logic_error("unhandled trim backend process status");
}

}  // namespace fa_trim::backends

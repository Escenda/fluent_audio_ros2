#include "fa_crossfade/backends/internal_crossfade.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_crossfade::backends
{

namespace
{
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;
constexpr double kHalfPi = 1.57079632679489661923;

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

void fadeGains(FadeCurve curve, size_t overlap_index, size_t overlap_frames, double & a_gain, double & b_gain)
{
  const double position =
    static_cast<double>(overlap_index + 1U) / static_cast<double>(overlap_frames + 1U);
  switch (curve) {
    case FadeCurve::kLinear:
      a_gain = 1.0 - position;
      b_gain = position;
      return;
    case FadeCurve::kEqualPower:
      a_gain = std::cos(position * kHalfPi);
      b_gain = std::sin(position * kHalfPi);
      return;
  }
  throw std::logic_error("unhandled crossfade fade curve");
}

ProcessStatus validateInputSamples(const std::vector<uint8_t> & input)
{
  const size_t sample_count = input.size() / sizeof(float);
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const float sample = readFloat32Le(input, sample_index);
    if (!std::isfinite(sample)) {
      return ProcessStatus::kNonFiniteInput;
    }
    if (!isNormalizedSample(sample)) {
      return ProcessStatus::kOutOfRangeInput;
    }
  }
  return ProcessStatus::kOk;
}
}  // namespace

InternalCrossfadeBackend::InternalCrossfadeBackend(const InternalCrossfadeConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.overlap_frames == 0U) {
    throw std::runtime_error("crossfade.overlap_frames must be > 0");
  }
  static_cast<void>(fadeCurveName(config_.fade_curve));
}

int InternalCrossfadeBackend::channels() const
{
  return config_.channels;
}

size_t InternalCrossfadeBackend::overlapFrames() const
{
  return config_.overlap_frames;
}

FadeCurve InternalCrossfadeBackend::fadeCurve() const
{
  return config_.fade_curve;
}

ProcessResult InternalCrossfadeBackend::process(
  const std::vector<uint8_t> & segment_a,
  const std::vector<uint8_t> & segment_b,
  std::vector<uint8_t> & output) const
{
  if (segment_a.empty() || segment_b.empty()) {
    return ProcessResult{ProcessStatus::kEmptyInput, 0U};
  }

  const size_t channels = static_cast<size_t>(config_.channels);
  const size_t bytes_per_frame = channels * sizeof(float);
  if ((segment_a.size() % bytes_per_frame) != 0 || (segment_b.size() % bytes_per_frame) != 0) {
    return ProcessResult{ProcessStatus::kMisalignedInput, 0U};
  }

  const size_t frames_a = segment_a.size() / bytes_per_frame;
  const size_t frames_b = segment_b.size() / bytes_per_frame;
  if (frames_a < config_.overlap_frames || frames_b < config_.overlap_frames) {
    return ProcessResult{ProcessStatus::kInputTooShort, 0U};
  }

  const ProcessStatus a_status = validateInputSamples(segment_a);
  if (a_status != ProcessStatus::kOk) {
    return ProcessResult{a_status, 0U};
  }
  const ProcessStatus b_status = validateInputSamples(segment_b);
  if (b_status != ProcessStatus::kOk) {
    return ProcessResult{b_status, 0U};
  }

  const size_t output_frames = frames_a + frames_b - config_.overlap_frames;
  std::vector<uint8_t> candidate(output_frames * bytes_per_frame, 0U);
  size_t output_sample_index = 0U;

  const size_t prefix_frames = frames_a - config_.overlap_frames;
  for (size_t frame_index = 0; frame_index < prefix_frames; ++frame_index) {
    for (size_t channel_index = 0; channel_index < channels; ++channel_index) {
      const size_t input_sample_index = frame_index * channels + channel_index;
      writeFloat32Le(candidate, output_sample_index, readFloat32Le(segment_a, input_sample_index));
      ++output_sample_index;
    }
  }

  for (size_t overlap_index = 0; overlap_index < config_.overlap_frames; ++overlap_index) {
    double a_gain = 0.0;
    double b_gain = 0.0;
    fadeGains(config_.fade_curve, overlap_index, config_.overlap_frames, a_gain, b_gain);
    for (size_t channel_index = 0; channel_index < channels; ++channel_index) {
      const size_t a_sample_index = (prefix_frames + overlap_index) * channels + channel_index;
      const size_t b_sample_index = overlap_index * channels + channel_index;
      const double output_sample =
        a_gain * static_cast<double>(readFloat32Le(segment_a, a_sample_index)) +
        b_gain * static_cast<double>(readFloat32Le(segment_b, b_sample_index));
      if (!std::isfinite(output_sample)) {
        return ProcessResult{ProcessStatus::kNonFiniteOutput, 0U};
      }
      const float output_float = static_cast<float>(output_sample);
      if (!isNormalizedSample(output_float)) {
        return ProcessResult{ProcessStatus::kOutOfRangeOutput, 0U};
      }
      writeFloat32Le(candidate, output_sample_index, output_float);
      ++output_sample_index;
    }
  }

  for (size_t frame_index = config_.overlap_frames; frame_index < frames_b; ++frame_index) {
    for (size_t channel_index = 0; channel_index < channels; ++channel_index) {
      const size_t input_sample_index = frame_index * channels + channel_index;
      writeFloat32Le(candidate, output_sample_index, readFloat32Le(segment_b, input_sample_index));
      ++output_sample_index;
    }
  }

  output = std::move(candidate);
  return ProcessResult{ProcessStatus::kOk, output_frames};
}

FadeCurve fadeCurveFromName(const std::string & name)
{
  if (name == "linear") {
    return FadeCurve::kLinear;
  }
  if (name == "equal_power") {
    return FadeCurve::kEqualPower;
  }
  throw std::logic_error("crossfade.curve must be one of: linear, equal_power");
}

const char * fadeCurveName(FadeCurve curve)
{
  switch (curve) {
    case FadeCurve::kLinear:
      return "linear";
    case FadeCurve::kEqualPower:
      return "equal_power";
  }
  throw std::logic_error("unhandled crossfade fade curve");
}

const char * processStatusMessage(ProcessStatus status)
{
  switch (status) {
    case ProcessStatus::kOk:
      return "ok";
    case ProcessStatus::kEmptyInput:
      return "input segment is empty";
    case ProcessStatus::kMisalignedInput:
      return "input byte length is not aligned to FLOAT32LE interleaved frames";
    case ProcessStatus::kInputTooShort:
      return "input segment is shorter than crossfade overlap";
    case ProcessStatus::kNonFiniteInput:
      return "input sample is not finite";
    case ProcessStatus::kOutOfRangeInput:
      return "input sample is outside normalized FLOAT32LE range";
    case ProcessStatus::kNonFiniteOutput:
      return "computed crossfade sample is not finite";
    case ProcessStatus::kOutOfRangeOutput:
      return "computed crossfade sample is outside normalized FLOAT32LE range";
  }
  throw std::logic_error("unhandled crossfade backend process status");
}

}  // namespace fa_crossfade::backends

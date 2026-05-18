#include "fa_declick/backends/internal_impulse_declick.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__) || \
  (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "fa_declick internal_impulse_declick requires a little-endian target for FLOAT32LE"
#endif

namespace fa_declick::backends
{

namespace
{
constexpr double kNormalizedMin = -1.0;
constexpr double kNormalizedMax = 1.0;

bool isFinite(const double value)
{
  return std::isfinite(value);
}

bool isNormalized(const double value)
{
  return isFinite(value) && value >= kNormalizedMin && value <= kNormalizedMax;
}

size_t sampleIndex(const size_t frame_index, const size_t channel_index, const size_t channel_count)
{
  return (frame_index * channel_count) + channel_index;
}
}  // namespace

InternalImpulseDeclickBackend::InternalImpulseDeclickBackend(
  const InternalImpulseDeclickConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (!isFinite(config_.threshold_delta) ||
      config_.threshold_delta <= 0.0 ||
      config_.threshold_delta > 2.0)
  {
    throw std::runtime_error("threshold.delta must be finite and in (0.0, 2.0]");
  }
  if (config_.max_click_samples <= 0) {
    throw std::runtime_error("window.max_samples must be > 0");
  }
}

ProcessResult InternalImpulseDeclickBackend::process(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output) const
{
  const size_t channel_count = static_cast<size_t>(config_.channels);
  const size_t bytes_per_frame = channel_count * sizeof(float);
  if (input.empty()) {
    return ProcessResult{ProcessStatus::kEmptyInput};
  }
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessResult{ProcessStatus::kMisalignedInput};
  }

  const size_t sample_count = input.size() / sizeof(float);
  const size_t frame_count = sample_count / channel_count;
  std::vector<float> input_samples(sample_count, 0.0F);
  std::vector<float> output_samples(sample_count, 0.0F);

  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    float sample = 0.0F;
    std::memcpy(&sample, input.data() + (sample_index * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteInput};
    }
    if (!isNormalized(static_cast<double>(sample))) {
      return ProcessResult{ProcessStatus::kOutOfRangeInput};
    }
    input_samples.at(sample_index) = sample;
    output_samples.at(sample_index) = sample;
  }

  uint64_t corrected_samples = 0;
  uint64_t corrected_runs = 0;
  if (frame_count >= 3) {
    for (size_t channel_index = 0; channel_index < channel_count; ++channel_index) {
      size_t frame_index = 1;
      while (frame_index + 1 < frame_count) {
        const size_t run_length = detectClickRun(
          input_samples, frame_index, channel_index, frame_count, channel_count);
        if (run_length == 0) {
          ++frame_index;
          continue;
        }

        const float previous = input_samples.at(
          sampleIndex(frame_index - 1, channel_index, channel_count));
        const float next = input_samples.at(
          sampleIndex(frame_index + run_length, channel_index, channel_count));
        const double corrected = (static_cast<double>(previous) + static_cast<double>(next)) / 2.0;
        if (!isNormalized(corrected)) {
          return ProcessResult{ProcessStatus::kOutOfRangeOutput};
        }

        const float corrected_sample = static_cast<float>(corrected);
        if (!std::isfinite(corrected_sample)) {
          return ProcessResult{ProcessStatus::kNonFiniteOutput};
        }
        if (!isNormalized(static_cast<double>(corrected_sample))) {
          return ProcessResult{ProcessStatus::kOutOfRangeOutput};
        }

        for (size_t offset = 0; offset < run_length; ++offset) {
          output_samples.at(sampleIndex(frame_index + offset, channel_index, channel_count)) =
            corrected_sample;
        }
        corrected_samples += run_length;
        ++corrected_runs;
        frame_index += run_length;
      }
    }
  }

  std::vector<uint8_t> next_output(input.size());
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const float sample = output_samples.at(sample_index);
    std::memcpy(next_output.data() + (sample_index * sizeof(float)), &sample, sizeof(float));
  }

  output = std::move(next_output);
  return ProcessResult{ProcessStatus::kOk, corrected_samples, corrected_runs};
}

size_t InternalImpulseDeclickBackend::detectClickRun(
  const std::vector<float> & samples,
  const size_t frame_index,
  const size_t channel_index,
  const size_t frame_count,
  const size_t channel_count) const
{
  if (frame_index == 0 || frame_index + 1 >= frame_count) {
    return 0;
  }

  const double delta = config_.threshold_delta;
  const size_t max_window = std::min(
    static_cast<size_t>(config_.max_click_samples),
    frame_count - frame_index - 1);
  const float previous = samples.at(sampleIndex(frame_index - 1, channel_index, channel_count));
  for (size_t run_length = 1; run_length <= max_window; ++run_length) {
    const float next = samples.at(sampleIndex(frame_index + run_length, channel_index, channel_count));
    if (std::abs(static_cast<double>(previous) - static_cast<double>(next)) > delta) {
      continue;
    }

    bool all_samples_are_clicks = true;
    for (size_t offset = 0; offset < run_length; ++offset) {
      const float current = samples.at(sampleIndex(frame_index + offset, channel_index, channel_count));
      const bool current_differs_from_previous =
        std::abs(static_cast<double>(current) - static_cast<double>(previous)) > delta;
      const bool current_differs_from_next =
        std::abs(static_cast<double>(current) - static_cast<double>(next)) > delta;
      if (!current_differs_from_previous || !current_differs_from_next) {
        all_samples_are_clicks = false;
        break;
      }
    }

    if (all_samples_are_clicks) {
      return run_length;
    }
  }

  return 0;
}

const char * processStatusMessage(const ProcessStatus status)
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
      return "input sample is outside normalized FLOAT32LE range [-1, 1]";
    case ProcessStatus::kNonFiniteOutput:
      return "output sample is not finite FLOAT32LE";
    case ProcessStatus::kOutOfRangeOutput:
      return "output sample is outside normalized FLOAT32LE range [-1, 1]";
  }
  return "unknown declick backend status";
}

}  // namespace fa_declick::backends

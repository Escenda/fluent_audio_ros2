#include "fa_silence_removal/backends/internal_rms_silence_removal.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace fa_silence_removal::backends
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

InternalRmsSilenceRemovalBackend::InternalRmsSilenceRemovalBackend(
  const InternalRmsSilenceRemovalConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (!std::isfinite(config_.threshold_rms) ||
      config_.threshold_rms < 0.0 ||
      config_.threshold_rms > 1.0)
  {
    throw std::runtime_error("threshold.rms must be finite and in [0.0, 1.0]");
  }
}

int InternalRmsSilenceRemovalBackend::channels() const
{
  return config_.channels;
}

double InternalRmsSilenceRemovalBackend::thresholdRms() const
{
  return config_.threshold_rms;
}

size_t InternalRmsSilenceRemovalBackend::hangoverSamples() const
{
  return config_.hangover_samples;
}

size_t InternalRmsSilenceRemovalBackend::hangoverSamplesRemaining() const
{
  return hangover_samples_remaining_;
}

double InternalRmsSilenceRemovalBackend::lastRms() const
{
  return last_rms_;
}

void InternalRmsSilenceRemovalBackend::consumeHangoverSamples(size_t frame_count)
{
  if (frame_count >= hangover_samples_remaining_) {
    hangover_samples_remaining_ = 0U;
    return;
  }
  hangover_samples_remaining_ -= frame_count;
}

ProcessResult InternalRmsSilenceRemovalBackend::process(const std::vector<uint8_t> & input)
{
  if (input.empty()) {
    return ProcessResult{
      ProcessStatus::kEmptyInput,
      Decision::kDroppedSilent,
      last_rms_,
      0U,
      hangover_samples_remaining_};
  }

  const size_t channels = static_cast<size_t>(config_.channels);
  const size_t bytes_per_frame = channels * sizeof(float);
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessResult{
      ProcessStatus::kMisalignedInput,
      Decision::kDroppedSilent,
      last_rms_,
      0U,
      hangover_samples_remaining_};
  }

  const size_t frame_count = input.size() / bytes_per_frame;
  const size_t sample_count = input.size() / sizeof(float);
  double sum_squares = 0.0;
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const float sample = readFloat32Le(input, sample_index);
    if (!std::isfinite(sample)) {
      return ProcessResult{
        ProcessStatus::kNonFiniteInput,
        Decision::kDroppedSilent,
        last_rms_,
        frame_count,
        hangover_samples_remaining_};
    }
    if (!isNormalizedSample(sample)) {
      return ProcessResult{
        ProcessStatus::kOutOfRangeInput,
        Decision::kDroppedSilent,
        last_rms_,
        frame_count,
        hangover_samples_remaining_};
    }

    const double sample_value = static_cast<double>(sample);
    sum_squares += sample_value * sample_value;
  }

  const double rms = std::sqrt(sum_squares / static_cast<double>(sample_count));
  if (!std::isfinite(rms) || rms < 0.0 || rms > 1.0) {
    return ProcessResult{
      ProcessStatus::kInvalidRms,
      Decision::kDroppedSilent,
      last_rms_,
      frame_count,
      hangover_samples_remaining_};
  }

  last_rms_ = rms;
  if (rms >= config_.threshold_rms) {
    hangover_samples_remaining_ = config_.hangover_samples;
    return ProcessResult{
      ProcessStatus::kOk,
      Decision::kAcceptedActive,
      rms,
      frame_count,
      hangover_samples_remaining_};
  }

  if (hangover_samples_remaining_ > 0U) {
    consumeHangoverSamples(frame_count);
    return ProcessResult{
      ProcessStatus::kOk,
      Decision::kAcceptedHangover,
      rms,
      frame_count,
      hangover_samples_remaining_};
  }

  return ProcessResult{
    ProcessStatus::kOk,
    Decision::kDroppedSilent,
    rms,
    frame_count,
    hangover_samples_remaining_};
}

const char * decisionName(Decision decision)
{
  switch (decision) {
    case Decision::kAcceptedActive:
      return "accepted_active";
    case Decision::kAcceptedHangover:
      return "accepted_hangover";
    case Decision::kDroppedSilent:
      return "dropped_silent";
  }
  throw std::logic_error("unhandled silence removal decision");
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
    case ProcessStatus::kInvalidRms:
      return "computed RMS is outside normalized range";
  }
  throw std::logic_error("unhandled silence removal backend process status");
}

}  // namespace fa_silence_removal::backends

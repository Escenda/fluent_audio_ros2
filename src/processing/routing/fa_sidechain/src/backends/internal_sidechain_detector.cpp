#include "fa_sidechain/backends/internal_sidechain_detector.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_sidechain::backends
{

namespace
{
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;
constexpr double kMinControlGain = 0.0;
constexpr double kMaxControlGain = 4.0;

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

std::vector<uint8_t> float32LeBytes(float sample)
{
  uint32_t bits = 0U;
  std::memcpy(&bits, &sample, sizeof(bits));
  return {
    static_cast<uint8_t>(bits & 0xFFU),
    static_cast<uint8_t>((bits >> 8U) & 0xFFU),
    static_cast<uint8_t>((bits >> 16U) & 0xFFU),
    static_cast<uint8_t>((bits >> 24U) & 0xFFU)};
}
}  // namespace

InternalSidechainDetectorBackend::InternalSidechainDetectorBackend(
  const InternalSidechainDetectorConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (!isFinite(config_.threshold_rms) ||
      config_.threshold_rms <= 0.0 ||
      config_.threshold_rms > 1.0)
  {
    throw std::runtime_error("detector.threshold_rms must be finite and in (0.0, 1.0]");
  }
  if (!isFinite(config_.active_gain_db)) {
    throw std::runtime_error("detector.active_gain_db must be finite");
  }
  if (!isFinite(config_.inactive_gain_db)) {
    throw std::runtime_error("detector.inactive_gain_db must be finite");
  }

  active_gain_linear_ = dbToLinear(config_.active_gain_db);
  inactive_gain_linear_ = dbToLinear(config_.inactive_gain_db);
  if (!isFinite(active_gain_linear_) ||
      active_gain_linear_ < kMinControlGain ||
      active_gain_linear_ > kMaxControlGain)
  {
    throw std::runtime_error("detector.active_gain_db must resolve to finite linear gain in [0.0, 4.0]");
  }
  if (!isFinite(inactive_gain_linear_) ||
      inactive_gain_linear_ < kMinControlGain ||
      inactive_gain_linear_ > kMaxControlGain)
  {
    throw std::runtime_error("detector.inactive_gain_db must resolve to finite linear gain in [0.0, 4.0]");
  }
  last_gain_linear_ = inactive_gain_linear_;
}

int InternalSidechainDetectorBackend::channels() const
{
  return config_.channels;
}

double InternalSidechainDetectorBackend::thresholdRms() const
{
  return config_.threshold_rms;
}

double InternalSidechainDetectorBackend::activeGainDb() const
{
  return config_.active_gain_db;
}

double InternalSidechainDetectorBackend::inactiveGainDb() const
{
  return config_.inactive_gain_db;
}

double InternalSidechainDetectorBackend::activeGainLinear() const
{
  return active_gain_linear_;
}

double InternalSidechainDetectorBackend::inactiveGainLinear() const
{
  return inactive_gain_linear_;
}

double InternalSidechainDetectorBackend::lastRms() const
{
  return last_rms_;
}

double InternalSidechainDetectorBackend::lastGainLinear() const
{
  return last_gain_linear_;
}

bool InternalSidechainDetectorBackend::lastActive() const
{
  return last_active_;
}

ProcessStatus InternalSidechainDetectorBackend::validateAndMeasure(
  const std::vector<uint8_t> & input,
  double & rms,
  size_t & frame_count) const
{
  if (input.empty()) {
    return ProcessStatus::kEmptyInput;
  }

  const size_t channels = static_cast<size_t>(config_.channels);
  const size_t bytes_per_frame = channels * sizeof(float);
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessStatus::kMisalignedInput;
  }

  frame_count = input.size() / bytes_per_frame;
  const size_t sample_count = input.size() / sizeof(float);
  double square_sum = 0.0;
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const float sample = readFloat32Le(input, sample_index);
    if (!std::isfinite(sample)) {
      return ProcessStatus::kNonFiniteInput;
    }
    if (!isNormalizedSample(sample)) {
      return ProcessStatus::kOutOfRangeInput;
    }
    const double sample_value = static_cast<double>(sample);
    square_sum += sample_value * sample_value;
  }

  rms = std::sqrt(square_sum / static_cast<double>(sample_count));
  if (!isFinite(rms) || rms < 0.0 || rms > 1.0) {
    return ProcessStatus::kInvalidRms;
  }
  return ProcessStatus::kOk;
}

DetectionResult InternalSidechainDetectorBackend::detect(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & control_data)
{
  double rms = 0.0;
  size_t frame_count = 0U;
  const ProcessStatus status = validateAndMeasure(input, rms, frame_count);
  if (status != ProcessStatus::kOk) {
    return DetectionResult{status, last_rms_, last_gain_linear_, last_active_, frame_count};
  }

  const bool active = rms >= config_.threshold_rms;
  const double gain_linear = active ? active_gain_linear_ : inactive_gain_linear_;
  if (!isFinite(gain_linear) || gain_linear < kMinControlGain || gain_linear > kMaxControlGain) {
    return DetectionResult{ProcessStatus::kInvalidGain, rms, last_gain_linear_, active, frame_count};
  }

  const float gain_sample = static_cast<float>(gain_linear);
  if (!std::isfinite(gain_sample)) {
    return DetectionResult{ProcessStatus::kNonFiniteOutput, rms, last_gain_linear_, active, frame_count};
  }
  if (gain_sample < static_cast<float>(kMinControlGain) ||
      gain_sample > static_cast<float>(kMaxControlGain))
  {
    return DetectionResult{ProcessStatus::kOutOfRangeOutput, rms, last_gain_linear_, active, frame_count};
  }

  control_data = float32LeBytes(gain_sample);
  last_rms_ = rms;
  last_gain_linear_ = gain_linear;
  last_active_ = active;
  return DetectionResult{ProcessStatus::kOk, rms, gain_linear, active, frame_count};
}

double dbToLinear(double db)
{
  return std::pow(10.0, db / 20.0);
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
    case ProcessStatus::kInvalidGain:
      return "sidechain gain is not finite";
    case ProcessStatus::kNonFiniteOutput:
      return "control gain output sample is not finite";
    case ProcessStatus::kOutOfRangeOutput:
      return "control gain output sample is outside supported range";
  }
  throw std::logic_error("unhandled sidechain detector backend process status");
}

}  // namespace fa_sidechain::backends

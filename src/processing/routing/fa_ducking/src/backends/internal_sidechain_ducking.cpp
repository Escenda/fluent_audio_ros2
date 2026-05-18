#include "fa_ducking/backends/internal_sidechain_ducking.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_ducking::backends
{

namespace
{
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;
constexpr double kMillisecondsPerSecond = 1000.0;

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

InternalSidechainDuckingBackend::InternalSidechainDuckingBackend(
  const InternalSidechainDuckingConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (!isFinite(config_.sidechain_threshold_rms) ||
      config_.sidechain_threshold_rms <= 0.0 ||
      config_.sidechain_threshold_rms > 1.0)
  {
    throw std::runtime_error("sidechain.threshold_rms must be finite and in (0.0, 1.0]");
  }
  if (config_.sidechain_max_age_ns <= 0) {
    throw std::runtime_error("sidechain.max_age_ms must convert to a positive duration");
  }
  if (!isFinite(config_.ducking_gain_db) ||
      config_.ducking_gain_db < -96.0 ||
      config_.ducking_gain_db >= 0.0)
  {
    throw std::runtime_error("ducking.gain_db must be finite and in [-96.0, 0.0)");
  }
  ducking_gain_linear_ = dbToLinear(config_.ducking_gain_db);
  if (!isFinite(ducking_gain_linear_) || ducking_gain_linear_ <= 0.0 || ducking_gain_linear_ >= 1.0) {
    throw std::runtime_error("ducking.gain_db must resolve to a finite attenuation gain in (0.0, 1.0)");
  }
  if (!isFinite(config_.attack_ms) || config_.attack_ms <= 0.0) {
    throw std::runtime_error("ducking.attack_ms must be finite and > 0.0");
  }
  if (!isFinite(config_.release_ms) || config_.release_ms <= 0.0) {
    throw std::runtime_error("ducking.release_ms must be finite and > 0.0");
  }
}

int InternalSidechainDuckingBackend::channels() const
{
  return config_.channels;
}

int InternalSidechainDuckingBackend::sampleRate() const
{
  return config_.sample_rate;
}

double InternalSidechainDuckingBackend::sidechainThresholdRms() const
{
  return config_.sidechain_threshold_rms;
}

int64_t InternalSidechainDuckingBackend::sidechainMaxAgeNs() const
{
  return config_.sidechain_max_age_ns;
}

double InternalSidechainDuckingBackend::duckingGainDb() const
{
  return config_.ducking_gain_db;
}

double InternalSidechainDuckingBackend::duckingGainLinear() const
{
  return ducking_gain_linear_;
}

double InternalSidechainDuckingBackend::attackMs() const
{
  return config_.attack_ms;
}

double InternalSidechainDuckingBackend::releaseMs() const
{
  return config_.release_ms;
}

double InternalSidechainDuckingBackend::currentGain() const
{
  return current_gain_;
}

double InternalSidechainDuckingBackend::lastTargetGain() const
{
  return last_target_gain_;
}

double InternalSidechainDuckingBackend::lastSidechainRms() const
{
  return last_sidechain_rms_;
}

int64_t InternalSidechainDuckingBackend::lastSidechainAgeNs() const
{
  return last_sidechain_age_ns_;
}

bool InternalSidechainDuckingBackend::lastSidechainActive() const
{
  return last_sidechain_active_;
}

bool InternalSidechainDuckingBackend::hasSidechain() const
{
  return has_sidechain_;
}

ProcessStatus InternalSidechainDuckingBackend::validateAndMeasure(
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

SidechainResult InternalSidechainDuckingBackend::observeSidechain(
  const std::vector<uint8_t> & input,
  int64_t now_ns)
{
  double rms = 0.0;
  size_t frame_count = 0U;
  const ProcessStatus status = validateAndMeasure(input, rms, frame_count);
  if (status != ProcessStatus::kOk) {
    return SidechainResult{status, last_sidechain_rms_, frame_count};
  }

  has_sidechain_ = true;
  latest_sidechain_rms_ = rms;
  latest_sidechain_received_ns_ = now_ns;
  last_sidechain_rms_ = rms;
  return SidechainResult{ProcessStatus::kOk, rms, frame_count};
}

void InternalSidechainDuckingBackend::invalidateSidechain()
{
  has_sidechain_ = false;
  latest_sidechain_rms_ = 0.0;
  latest_sidechain_received_ns_ = 0;
  last_sidechain_rms_ = 0.0;
  last_sidechain_age_ns_ = -1;
  last_sidechain_active_ = false;
}

double InternalSidechainDuckingBackend::smoothingAlpha(double time_constant_ms, size_t frame_count) const
{
  const double frame_seconds =
    static_cast<double>(frame_count) / static_cast<double>(config_.sample_rate);
  const double time_constant_seconds = time_constant_ms / kMillisecondsPerSecond;
  return 1.0 - std::exp(-frame_seconds / time_constant_seconds);
}

double InternalSidechainDuckingBackend::smoothedGain(double target_gain, size_t frame_count) const
{
  const double time_constant_ms = target_gain < current_gain_ ? config_.attack_ms : config_.release_ms;
  const double alpha = smoothingAlpha(time_constant_ms, frame_count);
  const double next_gain = current_gain_ + (alpha * (target_gain - current_gain_));
  if (next_gain < ducking_gain_linear_) {
    return ducking_gain_linear_;
  }
  if (next_gain > 1.0) {
    return 1.0;
  }
  return next_gain;
}

ProgramResult InternalSidechainDuckingBackend::processProgram(
  const std::vector<uint8_t> & input,
  int64_t now_ns,
  std::vector<uint8_t> & output)
{
  double rms = 0.0;
  size_t frame_count = 0U;
  const ProcessStatus status = validateAndMeasure(input, rms, frame_count);
  if (status != ProcessStatus::kOk) {
    return ProgramResult{
      status,
      false,
      false,
      last_sidechain_rms_,
      last_sidechain_age_ns_,
      last_target_gain_,
      current_gain_,
      frame_count};
  }

  bool sidechain_stale = false;
  bool sidechain_active = false;
  int64_t sidechain_age_ns = -1;
  double sidechain_rms = 0.0;
  if (has_sidechain_) {
    sidechain_age_ns = now_ns - latest_sidechain_received_ns_;
    sidechain_rms = latest_sidechain_rms_;
    if (sidechain_age_ns < 0 || sidechain_age_ns > config_.sidechain_max_age_ns) {
      sidechain_stale = true;
    } else {
      sidechain_active = sidechain_rms >= config_.sidechain_threshold_rms;
    }
  }

  const double target_gain = sidechain_active ? ducking_gain_linear_ : 1.0;
  const double candidate_gain = smoothedGain(target_gain, frame_count);
  if (!isFinite(target_gain) || !isFinite(candidate_gain)) {
    return ProgramResult{
      ProcessStatus::kInvalidGain,
      sidechain_active,
      sidechain_stale,
      sidechain_rms,
      sidechain_age_ns,
      target_gain,
      current_gain_,
      frame_count};
  }

  const size_t sample_count = input.size() / sizeof(float);
  std::vector<uint8_t> candidate(input.size(), 0U);
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const double processed_sample =
      static_cast<double>(readFloat32Le(input, sample_index)) * candidate_gain;
    if (!isFinite(processed_sample)) {
      return ProgramResult{
        ProcessStatus::kNonFiniteOutput,
        sidechain_active,
        sidechain_stale,
        sidechain_rms,
        sidechain_age_ns,
        target_gain,
        current_gain_,
        frame_count};
    }
    const float output_sample = static_cast<float>(processed_sample);
    if (!isNormalizedSample(output_sample)) {
      return ProgramResult{
        ProcessStatus::kOutOfRangeOutput,
        sidechain_active,
        sidechain_stale,
        sidechain_rms,
        sidechain_age_ns,
        target_gain,
        current_gain_,
        frame_count};
    }
    writeFloat32Le(candidate, sample_index, output_sample);
  }

  output = std::move(candidate);
  current_gain_ = candidate_gain;
  last_target_gain_ = target_gain;
  last_sidechain_age_ns_ = sidechain_age_ns;
  last_sidechain_active_ = sidechain_active;
  return ProgramResult{
    ProcessStatus::kOk,
    sidechain_active,
    sidechain_stale,
    sidechain_rms,
    sidechain_age_ns,
    target_gain,
    candidate_gain,
    frame_count};
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
      return "ducking gain is not finite";
    case ProcessStatus::kNonFiniteOutput:
      return "ducking output sample is not finite";
    case ProcessStatus::kOutOfRangeOutput:
      return "ducking output sample is outside normalized FLOAT32LE range";
  }
  throw std::logic_error("unhandled sidechain ducking backend process status");
}

}  // namespace fa_ducking::backends

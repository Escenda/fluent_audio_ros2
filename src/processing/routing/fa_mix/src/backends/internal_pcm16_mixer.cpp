#include "fa_mix/backends/internal_pcm16_mixer.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace fa_mix::backends
{

namespace
{
constexpr double kPcm16NegativeScale = 32768.0;
constexpr double kPcm16PositiveScale = 32767.0;

bool isFinite(double value)
{
  return std::isfinite(value);
}

int16_t readPcm16Le(const std::vector<uint8_t> & data, size_t sample_index)
{
  const size_t offset = sample_index * sizeof(int16_t);
  const uint16_t bits =
    static_cast<uint16_t>(data[offset]) |
    static_cast<uint16_t>(static_cast<uint16_t>(data[offset + 1U]) << 8U);
  return static_cast<int16_t>(bits);
}

void writePcm16Le(std::vector<uint8_t> & data, size_t sample_index, int16_t sample)
{
  const auto unsigned_sample = static_cast<uint16_t>(sample);
  const size_t offset = sample_index * sizeof(int16_t);
  data[offset] = static_cast<uint8_t>(unsigned_sample & 0x00FFU);
  data[offset + 1U] = static_cast<uint8_t>((unsigned_sample >> 8U) & 0x00FFU);
}
}  // namespace

InternalPcm16MixerBackend::InternalPcm16MixerBackend(const InternalPcm16MixerConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.input_gains_db.empty()) {
    throw std::runtime_error("input_gains_db must not be empty");
  }

  input_gains_linear_.reserve(config_.input_gains_db.size());
  for (double gain_db : config_.input_gains_db) {
    const double gain_linear = dbToLinear(gain_db);
    if (!isFinite(gain_db) || !isFinite(gain_linear)) {
      throw std::runtime_error("input_gains_db must resolve to finite linear gains");
    }
    input_gains_linear_.push_back(gain_linear);
  }
}

int InternalPcm16MixerBackend::channels() const
{
  return config_.channels;
}

size_t InternalPcm16MixerBackend::inputCount() const
{
  return input_gains_linear_.size();
}

const std::vector<double> & InternalPcm16MixerBackend::inputGainsDb() const
{
  return config_.input_gains_db;
}

const std::vector<double> & InternalPcm16MixerBackend::inputGainsLinear() const
{
  return input_gains_linear_;
}

size_t InternalPcm16MixerBackend::lastSampleCount() const
{
  return last_sample_count_;
}

MixStatus InternalPcm16MixerBackend::decodePcm16Le(
  const std::vector<uint8_t> & input,
  std::vector<float> & samples,
  size_t & frame_count) const
{
  samples.clear();
  frame_count = 0U;
  if (input.empty()) {
    return MixStatus::kEmptyInput;
  }

  const size_t bytes_per_frame = static_cast<size_t>(config_.channels) * sizeof(int16_t);
  if ((input.size() % bytes_per_frame) != 0) {
    return MixStatus::kMisalignedInput;
  }

  frame_count = input.size() / bytes_per_frame;
  const size_t sample_count = input.size() / sizeof(int16_t);
  samples.reserve(sample_count);
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const int16_t pcm_sample = readPcm16Le(input, sample_index);
    samples.push_back(static_cast<float>(static_cast<double>(pcm_sample) / kPcm16NegativeScale));
  }
  return MixStatus::kOk;
}

MixStatus InternalPcm16MixerBackend::encodePcm16Le(
  const std::vector<float> & samples,
  std::vector<uint8_t> & output) const
{
  if (samples.empty()) {
    return MixStatus::kEmptyInput;
  }

  std::vector<uint8_t> candidate(samples.size() * sizeof(int16_t), 0U);
  for (size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
    const float sample = samples[sample_index];
    if (!std::isfinite(sample)) {
      return MixStatus::kNonFiniteOutput;
    }
    if (sample < -1.0F || sample > 1.0F) {
      return MixStatus::kOutOfRangeOutput;
    }

    const double scaled = sample < 0.0F ?
      static_cast<double>(sample) * kPcm16NegativeScale :
      static_cast<double>(sample) * kPcm16PositiveScale;
    const int32_t rounded = static_cast<int32_t>(std::lround(scaled));
    if (rounded < -32768 || rounded > 32767) {
      return MixStatus::kPcm16RangeOutput;
    }
    writePcm16Le(candidate, sample_index, static_cast<int16_t>(rounded));
  }

  output = std::move(candidate);
  return MixStatus::kOk;
}

MixResult InternalPcm16MixerBackend::mix(
  const std::vector<std::vector<uint8_t>> & inputs,
  std::vector<uint8_t> & output)
{
  if (inputs.empty()) {
    return MixResult{MixStatus::kNoInputs, 0U, 0U};
  }
  if (inputs.size() != input_gains_linear_.size()) {
    return MixResult{MixStatus::kInputCountMismatch, inputs.size(), 0U};
  }

  std::vector<float> mixed;
  size_t expected_sample_count = 0U;
  for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
    std::vector<float> decoded;
    size_t frame_count = 0U;
    const MixStatus decode_status = decodePcm16Le(inputs[input_index], decoded, frame_count);
    if (decode_status != MixStatus::kOk) {
      return MixResult{decode_status, inputs.size(), expected_sample_count};
    }

    if (input_index == 0U) {
      mixed.assign(decoded.size(), 0.0F);
      expected_sample_count = decoded.size();
    } else if (decoded.size() != expected_sample_count) {
      return MixResult{MixStatus::kSampleCountMismatch, inputs.size(), decoded.size()};
    }

    const double gain = input_gains_linear_[input_index];
    if (!isFinite(gain)) {
      return MixResult{MixStatus::kNonFiniteGain, inputs.size(), expected_sample_count};
    }
    for (size_t sample_index = 0; sample_index < expected_sample_count; ++sample_index) {
      mixed[sample_index] += static_cast<float>(static_cast<double>(decoded[sample_index]) * gain);
    }
  }

  std::vector<uint8_t> candidate;
  const MixStatus encode_status = encodePcm16Le(mixed, candidate);
  if (encode_status != MixStatus::kOk) {
    return MixResult{encode_status, inputs.size(), expected_sample_count};
  }

  output = std::move(candidate);
  last_sample_count_ = expected_sample_count;
  return MixResult{MixStatus::kOk, inputs.size(), expected_sample_count};
}

double dbToLinear(double db)
{
  return std::pow(10.0, db / 20.0);
}

const char * mixStatusMessage(MixStatus status)
{
  switch (status) {
    case MixStatus::kOk:
      return "ok";
    case MixStatus::kNoInputs:
      return "no input buffers were provided";
    case MixStatus::kInputCountMismatch:
      return "input buffer count does not match configured gain count";
    case MixStatus::kEmptyInput:
      return "input data is empty";
    case MixStatus::kMisalignedInput:
      return "input byte length is not aligned to PCM16LE interleaved frames";
    case MixStatus::kSampleCountMismatch:
      return "input sample counts differ";
    case MixStatus::kNonFiniteGain:
      return "input gain is not finite";
    case MixStatus::kNonFiniteOutput:
      return "mixed sample is not finite";
    case MixStatus::kOutOfRangeOutput:
      return "mixed sample out of normalized PCM16 range";
    case MixStatus::kPcm16RangeOutput:
      return "mixed sample does not fit PCM16 after scaling";
  }
  throw std::logic_error("unhandled PCM16 mixer backend status");
}

}  // namespace fa_mix::backends

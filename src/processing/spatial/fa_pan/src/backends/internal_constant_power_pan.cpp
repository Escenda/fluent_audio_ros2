#include "fa_pan/backends/internal_constant_power_pan.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_pan::backends
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;
constexpr size_t kStereoChannels = 2U;

bool isFinite(double value)
{
  return std::isfinite(value);
}
}  // namespace

InternalConstantPowerPanBackend::InternalConstantPowerPanBackend(
  const InternalConstantPowerPanConfig & config)
{
  if (!isFinite(config.position) || config.position < -1.0 || config.position > 1.0) {
    throw std::runtime_error("pan.position must be finite and in [-1.0, 1.0]");
  }

  const double angle = (config.position + 1.0) * kPi / 4.0;
  left_gain_ = std::cos(angle);
  right_gain_ = std::sin(angle);

  if (!isFinite(left_gain_) || !isFinite(right_gain_)) {
    throw std::runtime_error("pan gain calculation produced non-finite coefficients");
  }
}

double InternalConstantPowerPanBackend::leftGain() const
{
  return left_gain_;
}

double InternalConstantPowerPanBackend::rightGain() const
{
  return right_gain_;
}

ProcessStatus InternalConstantPowerPanBackend::process(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output) const
{
  if (input.empty() || (input.size() % (kStereoChannels * sizeof(float))) != 0) {
    return ProcessStatus::kInvalidInputSize;
  }

  std::vector<uint8_t> next_output;
  next_output.reserve(input.size());

  const size_t sample_count = input.size() / sizeof(float);
  for (size_t sample_index = 0; sample_index < sample_count; sample_index += kStereoChannels) {
    const float left = readFloat32Le(input, sample_index);
    const float right = readFloat32Le(input, sample_index + 1U);
    if (!isNormalizedFinite(left) || !isNormalizedFinite(right)) {
      return ProcessStatus::kInvalidInputSample;
    }

    const double panned_left = static_cast<double>(left) * left_gain_;
    const double panned_right = static_cast<double>(right) * right_gain_;
    if (!isFinite(panned_left) || !isFinite(panned_right) ||
        panned_left < kMinNormalizedSample || panned_left > kMaxNormalizedSample ||
        panned_right < kMinNormalizedSample || panned_right > kMaxNormalizedSample)
    {
      return ProcessStatus::kInvalidOutputSample;
    }

    appendFloat32Le(static_cast<float>(panned_left), next_output);
    appendFloat32Le(static_cast<float>(panned_right), next_output);
  }

  output = std::move(next_output);
  return ProcessStatus::kOk;
}

const char * processStatusMessage(ProcessStatus status)
{
  switch (status) {
    case ProcessStatus::kOk:
      return "ok";
    case ProcessStatus::kInvalidInputSize:
      return "invalid stereo FLOAT32LE input size";
    case ProcessStatus::kInvalidInputSample:
      return "input sample is outside normalized FLOAT32LE range";
    case ProcessStatus::kInvalidOutputSample:
      return "pan output is outside normalized FLOAT32LE range";
  }
  throw std::logic_error("unhandled internal_constant_power_pan status");
}

bool isNormalizedFinite(float sample)
{
  return std::isfinite(sample) && sample >= kMinNormalizedSample && sample <= kMaxNormalizedSample;
}

float readFloat32Le(const std::vector<uint8_t> & bytes, size_t sample_index)
{
  uint32_t raw =
    static_cast<uint32_t>(bytes.at(sample_index * sizeof(float))) |
    (static_cast<uint32_t>(bytes.at((sample_index * sizeof(float)) + 1U)) << 8U) |
    (static_cast<uint32_t>(bytes.at((sample_index * sizeof(float)) + 2U)) << 16U) |
    (static_cast<uint32_t>(bytes.at((sample_index * sizeof(float)) + 3U)) << 24U);
  float sample = 0.0F;
  std::memcpy(&sample, &raw, sizeof(float));
  return sample;
}

void appendFloat32Le(float sample, std::vector<uint8_t> & out_bytes)
{
  uint32_t raw = 0;
  std::memcpy(&raw, &sample, sizeof(float));
  out_bytes.push_back(static_cast<uint8_t>(raw & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 8U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 16U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 24U) & 0xFFU));
}

}  // namespace fa_pan::backends

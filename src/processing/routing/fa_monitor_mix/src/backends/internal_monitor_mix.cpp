#include "fa_monitor_mix/backends/internal_monitor_mix.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_monitor_mix::backends
{

namespace
{
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;
constexpr size_t kFloat32Bytes = sizeof(float);

bool isFinite(const double value)
{
  return std::isfinite(value);
}
}  // namespace

InternalMonitorMixBackend::InternalMonitorMixBackend(const InternalMonitorMixConfig & config)
: config_(config)
{
  if (config_.input_count == 0) {
    throw std::runtime_error("monitor mix backend requires at least one input");
  }
  if (config_.master_index >= config_.input_count) {
    throw std::runtime_error("monitor mix backend master_index out of range");
  }
  if (config_.channels == 0) {
    throw std::runtime_error("monitor mix backend channels must be > 0");
  }
  if (config_.gains_linear.size() != config_.input_count) {
    throw std::runtime_error("monitor mix backend gains must match input count");
  }
  for (const double gain : config_.gains_linear) {
    if (!isFinite(gain)) {
      throw std::runtime_error("monitor mix backend gains must be finite");
    }
  }
}

ProcessStatus InternalMonitorMixBackend::validateFrameBytes(const std::vector<uint8_t> & data) const
{
  if (data.empty()) {
    return ProcessStatus::kEmptyInput;
  }
  const size_t bytes_per_frame = config_.channels * kFloat32Bytes;
  if (bytes_per_frame == 0 || (data.size() % bytes_per_frame) != 0) {
    return ProcessStatus::kMisalignedInput;
  }

  std::vector<float> samples;
  return decodeSamples(data, samples);
}

ProcessResult InternalMonitorMixBackend::mix(const std::vector<std::vector<uint8_t>> & inputs) const
{
  ProcessResult result;
  if (inputs.size() != config_.input_count) {
    result.status = ProcessStatus::kInputCountMismatch;
    return result;
  }

  const size_t expected_bytes = inputs[config_.master_index].size();
  std::vector<float> mixed;
  ProcessStatus status = decodeSamples(inputs[config_.master_index], mixed);
  if (status != ProcessStatus::kOk) {
    result.status = status;
    return result;
  }

  const float master_gain = static_cast<float>(config_.gains_linear[config_.master_index]);
  for (float & sample : mixed) {
    sample *= master_gain;
    if (!std::isfinite(sample)) {
      result.status = ProcessStatus::kNonFiniteOutput;
      return result;
    }
    if (sample < kMinNormalizedSample || sample > kMaxNormalizedSample) {
      result.status = ProcessStatus::kOutOfRangeOutput;
      return result;
    }
  }

  for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
    if (input_index == config_.master_index) {
      continue;
    }
    if (inputs[input_index].size() != expected_bytes) {
      result.status = ProcessStatus::kByteLengthMismatch;
      return result;
    }

    std::vector<float> input_samples;
    status = decodeSamples(inputs[input_index], input_samples);
    if (status != ProcessStatus::kOk) {
      result.status = status;
      return result;
    }
    const float gain = static_cast<float>(config_.gains_linear[input_index]);
    for (size_t sample_index = 0; sample_index < mixed.size(); ++sample_index) {
      const float output = mixed[sample_index] + (input_samples[sample_index] * gain);
      if (!std::isfinite(output)) {
        result.status = ProcessStatus::kNonFiniteOutput;
        return result;
      }
      if (output < kMinNormalizedSample || output > kMaxNormalizedSample) {
        result.status = ProcessStatus::kOutOfRangeOutput;
        return result;
      }
      mixed[sample_index] = output;
    }
  }

  std::vector<uint8_t> output(mixed.size() * kFloat32Bytes);
  std::memcpy(output.data(), mixed.data(), output.size());
  result.status = ProcessStatus::kOk;
  result.output = std::move(output);
  return result;
}

ProcessStatus InternalMonitorMixBackend::decodeSamples(
  const std::vector<uint8_t> & data,
  std::vector<float> & samples) const
{
  samples.clear();
  if (data.empty()) {
    return ProcessStatus::kEmptyInput;
  }
  if ((data.size() % kFloat32Bytes) != 0) {
    return ProcessStatus::kMisalignedInput;
  }

  std::vector<float> next_samples(data.size() / kFloat32Bytes);
  std::memcpy(next_samples.data(), data.data(), data.size());
  for (const float sample : next_samples) {
    if (!std::isfinite(sample)) {
      return ProcessStatus::kNonFiniteInput;
    }
    if (sample < kMinNormalizedSample || sample > kMaxNormalizedSample) {
      return ProcessStatus::kOutOfRangeInput;
    }
  }
  samples = std::move(next_samples);
  return ProcessStatus::kOk;
}

const char * processStatusMessage(const ProcessStatus status)
{
  switch (status) {
    case ProcessStatus::kOk:
      return "ok";
    case ProcessStatus::kInputCountMismatch:
      return "input count does not match monitor mix backend config";
    case ProcessStatus::kEmptyInput:
      return "input frame bytes are empty";
    case ProcessStatus::kMisalignedInput:
      return "input frame bytes are not aligned to FLOAT32LE channel frames";
    case ProcessStatus::kByteLengthMismatch:
      return "input byte length differs from master";
    case ProcessStatus::kNonFiniteInput:
      return "input sample is not finite";
    case ProcessStatus::kOutOfRangeInput:
      return "input sample is outside normalized FLOAT32LE range";
    case ProcessStatus::kNonFiniteOutput:
      return "mixed output sample is not finite";
    case ProcessStatus::kOutOfRangeOutput:
      return "mixed output sample is outside normalized FLOAT32LE range";
  }
  return "unknown monitor mix backend status";
}

}  // namespace fa_monitor_mix::backends

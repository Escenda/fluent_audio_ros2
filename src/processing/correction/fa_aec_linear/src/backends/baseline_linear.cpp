#include "fa_aec_linear/backends/baseline_linear.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__) || \
  (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "fa_aec_linear baseline_linear requires a little-endian target for PCM16LE/FLOAT32LE"
#endif

namespace fa_aec_linear::backends
{

namespace
{
constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
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
}  // namespace

bool isSupportedAudioFormatPair(const std::string & encoding, const int bit_depth)
{
  return (encoding == kEncodingPcm16 && bit_depth == 16) ||
         (encoding == kEncodingFloat32 && bit_depth == 32);
}

BaselineLinearBackend::BaselineLinearBackend(const BaselineLinearConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected_channels must be > 0");
  }
  if (!isSupportedAudioFormatPair(config_.encoding, config_.bit_depth)) {
    throw std::runtime_error("expected encoding/bit_depth must be PCM16LE/16 or FLOAT32LE/32");
  }
  if (!isFinite(config_.cancel_gain)) {
    throw std::runtime_error("cancel_gain must be finite");
  }
}

ProcessResult BaselineLinearBackend::process(
  const std::vector<uint8_t> & mic_input,
  const std::vector<uint8_t> & reference_input,
  std::vector<uint8_t> & output) const
{
  std::vector<float> mic_samples;
  const ProcessStatus mic_status = decodeToFloat(mic_input, false, mic_samples);
  if (mic_status != ProcessStatus::kOk) {
    return ProcessResult{mic_status};
  }

  std::vector<float> ref_samples;
  const ProcessStatus ref_status = decodeToFloat(reference_input, true, ref_samples);
  if (ref_status != ProcessStatus::kOk) {
    return ProcessResult{ref_status};
  }

  if (mic_samples.size() != ref_samples.size()) {
    return ProcessResult{ProcessStatus::kSampleCountMismatch};
  }

  std::vector<float> output_samples(mic_samples.size(), 0.0F);
  for (size_t sample_index = 0; sample_index < mic_samples.size(); ++sample_index) {
    const double corrected =
      static_cast<double>(mic_samples.at(sample_index)) -
      (config_.cancel_gain * static_cast<double>(ref_samples.at(sample_index)));
    if (!isFinite(corrected)) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput};
    }
    if (!isNormalized(corrected)) {
      return ProcessResult{ProcessStatus::kOutOfRangeOutput};
    }

    const float output_sample = static_cast<float>(corrected);
    if (!std::isfinite(output_sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput};
    }
    if (!isNormalized(static_cast<double>(output_sample))) {
      return ProcessResult{ProcessStatus::kOutOfRangeOutput};
    }
    output_samples.at(sample_index) = output_sample;
  }

  std::vector<uint8_t> next_output;
  const ProcessStatus encode_status = encodeFromFloat(output_samples, next_output);
  if (encode_status != ProcessStatus::kOk) {
    return ProcessResult{encode_status};
  }

  output = std::move(next_output);
  return ProcessResult{ProcessStatus::kOk};
}

ProcessStatus BaselineLinearBackend::decodeToFloat(
  const std::vector<uint8_t> & input,
  const bool is_reference,
  std::vector<float> & samples) const
{
  samples.clear();
  if (input.empty()) {
    return is_reference ? ProcessStatus::kEmptyReference : ProcessStatus::kEmptyMic;
  }
  if ((input.size() % bytesPerFrame()) != 0) {
    return is_reference ? ProcessStatus::kMisalignedReference : ProcessStatus::kMisalignedMic;
  }

  const size_t sample_count = input.size() / bytesPerSample();
  if (sample_count == 0) {
    return is_reference ? ProcessStatus::kEmptyReference : ProcessStatus::kEmptyMic;
  }
  samples.resize(sample_count);

  if (config_.encoding == kEncodingPcm16 && config_.bit_depth == 16) {
    for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
      int16_t pcm = 0;
      std::memcpy(&pcm, input.data() + (sample_index * sizeof(int16_t)), sizeof(int16_t));
      samples.at(sample_index) = static_cast<float>(pcm) / 32768.0F;
    }
    return ProcessStatus::kOk;
  }

  if (config_.encoding != kEncodingFloat32 || config_.bit_depth != 32) {
    return ProcessStatus::kUnsupportedFormat;
  }

  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    float sample = 0.0F;
    std::memcpy(&sample, input.data() + (sample_index * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample)) {
      return is_reference ? ProcessStatus::kNonFiniteReference : ProcessStatus::kNonFiniteMic;
    }
    if (!isNormalized(static_cast<double>(sample))) {
      return is_reference ? ProcessStatus::kOutOfRangeReference : ProcessStatus::kOutOfRangeMic;
    }
    samples.at(sample_index) = sample;
  }
  return ProcessStatus::kOk;
}

ProcessStatus BaselineLinearBackend::encodeFromFloat(
  const std::vector<float> & samples,
  std::vector<uint8_t> & output) const
{
  if (samples.empty()) {
    return ProcessStatus::kEmptyMic;
  }

  if (config_.encoding == kEncodingPcm16 && config_.bit_depth == 16) {
    std::vector<uint8_t> bytes(samples.size() * sizeof(int16_t));
    for (size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
      const float sample = samples.at(sample_index);
      if (!std::isfinite(sample)) {
        return ProcessStatus::kNonFiniteOutput;
      }
      if (!isNormalized(static_cast<double>(sample))) {
        return ProcessStatus::kOutOfRangeOutput;
      }
      const double scaled = sample < 0.0F ?
        static_cast<double>(sample) * 32768.0 :
        static_cast<double>(sample) * 32767.0;
      const int32_t rounded = static_cast<int32_t>(std::lround(scaled));
      if (rounded < -32768 || rounded > 32767) {
        return ProcessStatus::kPcm16OutputOutOfRange;
      }
      const int16_t pcm = static_cast<int16_t>(rounded);
      std::memcpy(bytes.data() + (sample_index * sizeof(int16_t)), &pcm, sizeof(int16_t));
    }
    output = std::move(bytes);
    return ProcessStatus::kOk;
  }

  if (config_.encoding == kEncodingFloat32 && config_.bit_depth == 32) {
    std::vector<uint8_t> bytes(samples.size() * sizeof(float));
    for (size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
      const float sample = samples.at(sample_index);
      if (!std::isfinite(sample)) {
        return ProcessStatus::kNonFiniteOutput;
      }
      if (!isNormalized(static_cast<double>(sample))) {
        return ProcessStatus::kOutOfRangeOutput;
      }
      std::memcpy(bytes.data() + (sample_index * sizeof(float)), &sample, sizeof(float));
    }
    output = std::move(bytes);
    return ProcessStatus::kOk;
  }

  return ProcessStatus::kUnsupportedFormat;
}

size_t BaselineLinearBackend::bytesPerFrame() const
{
  return static_cast<size_t>(config_.channels) * bytesPerSample();
}

size_t BaselineLinearBackend::bytesPerSample() const
{
  return static_cast<size_t>(config_.bit_depth / 8);
}

const char * processStatusMessage(const ProcessStatus status)
{
  switch (status) {
    case ProcessStatus::kOk:
      return "ok";
    case ProcessStatus::kEmptyMic:
      return "mic input data is empty";
    case ProcessStatus::kEmptyReference:
      return "reference input data is empty";
    case ProcessStatus::kMisalignedMic:
      return "mic input byte length is not aligned to expected frames";
    case ProcessStatus::kMisalignedReference:
      return "reference input byte length is not aligned to expected frames";
    case ProcessStatus::kUnsupportedFormat:
      return "audio format must be PCM16LE/16 or FLOAT32LE/32";
    case ProcessStatus::kSampleCountMismatch:
      return "mic and reference sample counts differ";
    case ProcessStatus::kNonFiniteMic:
      return "mic FLOAT32 sample is not finite";
    case ProcessStatus::kNonFiniteReference:
      return "reference FLOAT32 sample is not finite";
    case ProcessStatus::kOutOfRangeMic:
      return "mic FLOAT32 sample is outside normalized range [-1, 1]";
    case ProcessStatus::kOutOfRangeReference:
      return "reference FLOAT32 sample is outside normalized range [-1, 1]";
    case ProcessStatus::kNonFiniteOutput:
      return "AEC linear output sample is not finite";
    case ProcessStatus::kOutOfRangeOutput:
      return "AEC linear output sample is outside normalized range [-1, 1]";
    case ProcessStatus::kPcm16OutputOutOfRange:
      return "AEC linear output sample does not fit PCM16 after scaling";
  }
  throw std::logic_error("unhandled AEC linear backend process status");
}

}  // namespace fa_aec_linear::backends

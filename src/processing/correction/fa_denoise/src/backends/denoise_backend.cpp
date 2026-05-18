#include "fa_denoise/backends/denoise_backend.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace fa_denoise::backends
{

namespace
{
ProcessStatus validateOutputSamples(const std::vector<float> & samples)
{
  if (samples.empty()) {
    return ProcessStatus::kEmptyOutput;
  }
  for (const float sample : samples) {
    if (!std::isfinite(sample)) {
      return ProcessStatus::kNonFiniteOutput;
    }
    if (sample < -1.0f || sample > 1.0f) {
      return ProcessStatus::kOutOfRangeOutput;
    }
  }
  return ProcessStatus::kOk;
}
}  // namespace

size_t DenoiseBackend::pendingInputSamples() const
{
  return 0;
}

bool isSupportedAudioFormatPair(const std::string & encoding, const int bit_depth)
{
  return (encoding == kEncodingPcm16 && bit_depth == 16) ||
         (encoding == kEncodingFloat32 && bit_depth == 32);
}

size_t bytesPerSample(const AudioFormat & format)
{
  if (!isSupportedAudioFormatPair(format.encoding, format.bit_depth)) {
    return 0;
  }
  return static_cast<size_t>(format.bit_depth / 8);
}

ProcessStatus validateAudioBuffer(const AudioBuffer & input)
{
  if (!isSupportedAudioFormatPair(input.format.encoding, input.format.bit_depth)) {
    return ProcessStatus::kUnsupportedInputFormat;
  }
  if (input.format.sample_rate <= 0 || input.format.channels <= 0) {
    return ProcessStatus::kUnsupportedInputFormat;
  }
  if (input.data.empty()) {
    return ProcessStatus::kEmptyInput;
  }
  const size_t sample_bytes = bytesPerSample(input.format);
  const size_t frame_bytes = static_cast<size_t>(input.format.channels) * sample_bytes;
  if (frame_bytes == 0 || (input.data.size() % frame_bytes) != 0) {
    return ProcessStatus::kMisalignedInput;
  }
  return ProcessStatus::kOk;
}

ProcessStatus decodeToFloat(const AudioBuffer & input, std::vector<float> & output)
{
  const ProcessStatus validation_status = validateAudioBuffer(input);
  if (validation_status != ProcessStatus::kOk) {
    return validation_status;
  }

  const size_t sample_bytes = bytesPerSample(input.format);
  const size_t sample_count = input.data.size() / sample_bytes;
  if (sample_count == 0) {
    return ProcessStatus::kEmptyInput;
  }

  std::vector<float> next_output(sample_count);
  if (input.format.encoding == kEncodingPcm16 && input.format.bit_depth == 16) {
    std::vector<int16_t> pcm(sample_count);
    std::memcpy(pcm.data(), input.data.data(), input.data.size());
    for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
      next_output[sample_index] = static_cast<float>(pcm[sample_index]) / 32768.0f;
    }
    output = std::move(next_output);
    return ProcessStatus::kOk;
  }

  if (input.format.encoding != kEncodingFloat32 || input.format.bit_depth != 32) {
    return ProcessStatus::kUnsupportedInputFormat;
  }
  std::memcpy(next_output.data(), input.data.data(), input.data.size());
  for (const float sample : next_output) {
    if (!std::isfinite(sample)) {
      return ProcessStatus::kNonFiniteInput;
    }
    if (sample < -1.0f || sample > 1.0f) {
      return ProcessStatus::kOutOfRangeInput;
    }
  }
  output = std::move(next_output);
  return ProcessStatus::kOk;
}

ProcessStatus encodeFromFloat(
  const std::vector<float> & samples,
  const AudioFormat & output_format,
  std::vector<uint8_t> & output_bytes)
{
  if (!isSupportedAudioFormatPair(output_format.encoding, output_format.bit_depth)) {
    return ProcessStatus::kUnsupportedOutputFormat;
  }
  if (output_format.sample_rate <= 0 || output_format.channels <= 0) {
    return ProcessStatus::kUnsupportedOutputFormat;
  }

  const ProcessStatus sample_status = validateOutputSamples(samples);
  if (sample_status != ProcessStatus::kOk) {
    return sample_status;
  }

  if (output_format.encoding == kEncodingPcm16 && output_format.bit_depth == 16) {
    std::vector<int16_t> pcm(samples.size());
    for (size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
      const double scaled = samples[sample_index] < 0.0f ?
        static_cast<double>(samples[sample_index]) * 32768.0 :
        static_cast<double>(samples[sample_index]) * 32767.0;
      const long rounded = std::lround(scaled);
      if (rounded < std::numeric_limits<int16_t>::min() ||
          rounded > std::numeric_limits<int16_t>::max())
      {
        return ProcessStatus::kPcm16OutputOutOfRange;
      }
      pcm[sample_index] = static_cast<int16_t>(rounded);
    }
    std::vector<uint8_t> next_output(pcm.size() * sizeof(int16_t));
    std::memcpy(next_output.data(), pcm.data(), next_output.size());
    output_bytes = std::move(next_output);
    return ProcessStatus::kOk;
  }

  if (output_format.encoding == kEncodingFloat32 && output_format.bit_depth == 32) {
    std::vector<uint8_t> next_output(samples.size() * sizeof(float));
    std::memcpy(next_output.data(), samples.data(), next_output.size());
    output_bytes = std::move(next_output);
    return ProcessStatus::kOk;
  }

  return ProcessStatus::kUnsupportedOutputFormat;
}

const char * processStatusMessage(const ProcessStatus status)
{
  switch (status) {
    case ProcessStatus::kOk:
      return "ok";
    case ProcessStatus::kEmptyInput:
      return "input audio buffer is empty";
    case ProcessStatus::kUnsupportedInputFormat:
      return "input format must be PCM16LE/16 or FLOAT32LE/32 with positive sample rate and channels";
    case ProcessStatus::kUnsupportedOutputFormat:
      return "output format must be PCM16LE/16 or FLOAT32LE/32 with positive sample rate and channels";
    case ProcessStatus::kMisalignedInput:
      return "input audio buffer is not aligned to complete frames";
    case ProcessStatus::kNonFiniteInput:
      return "input FLOAT32LE sample is not finite";
    case ProcessStatus::kOutOfRangeInput:
      return "input FLOAT32LE sample is outside normalized [-1.0, 1.0] range";
    case ProcessStatus::kSampleCountNotBlockAligned:
      return "input sample count is not aligned to dtln.block_shift";
    case ProcessStatus::kProcessingFailed:
      return "denoise backend processing failed";
    case ProcessStatus::kOutputSampleCountMismatch:
      return "denoise backend output sample count does not match input sample count";
    case ProcessStatus::kEmptyOutput:
      return "denoise output sample buffer is empty";
    case ProcessStatus::kNonFiniteOutput:
      return "denoise output sample is not finite";
    case ProcessStatus::kOutOfRangeOutput:
      return "denoise output sample is outside normalized [-1.0, 1.0] range";
    case ProcessStatus::kPcm16OutputOutOfRange:
      return "denoise output sample does not fit PCM16 after scaling";
    case ProcessStatus::kPassthroughFormatMismatch:
      return "passthrough backend requires output format to match expected input format";
  }
  return "unknown denoise backend status";
}

}  // namespace fa_denoise::backends

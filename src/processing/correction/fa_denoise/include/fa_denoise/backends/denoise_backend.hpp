#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_denoise::backends
{

constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kEncodingFloat32 = "FLOAT32LE";

struct AudioFormat
{
  int sample_rate = -1;
  int channels = -1;
  std::string encoding;
  int bit_depth = -1;
};

struct AudioBuffer
{
  AudioFormat format;
  std::vector<uint8_t> data;
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kUnsupportedInputFormat,
  kUnsupportedOutputFormat,
  kMisalignedInput,
  kNonFiniteInput,
  kOutOfRangeInput,
  kSampleCountNotBlockAligned,
  kProcessingFailed,
  kOutputSampleCountMismatch,
  kEmptyOutput,
  kNonFiniteOutput,
  kOutOfRangeOutput,
  kPcm16OutputOutOfRange,
  kPassthroughFormatMismatch,
};

struct ProcessResult
{
  ProcessStatus status = ProcessStatus::kOk;
  AudioBuffer output;
};

class DenoiseBackend
{
public:
  virtual ~DenoiseBackend() = default;

  [[nodiscard]] virtual const char * name() const = 0;
  [[nodiscard]] virtual ProcessResult process(const AudioBuffer & input) = 0;
  [[nodiscard]] virtual size_t pendingInputSamples() const;
};

[[nodiscard]] bool isSupportedAudioFormatPair(const std::string & encoding, int bit_depth);
[[nodiscard]] size_t bytesPerSample(const AudioFormat & format);
[[nodiscard]] ProcessStatus validateAudioBuffer(const AudioBuffer & input);
[[nodiscard]] ProcessStatus decodeToFloat(const AudioBuffer & input, std::vector<float> & output);
[[nodiscard]] ProcessStatus encodeFromFloat(
  const std::vector<float> & samples,
  const AudioFormat & output_format,
  std::vector<uint8_t> & output_bytes);
[[nodiscard]] const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_denoise::backends

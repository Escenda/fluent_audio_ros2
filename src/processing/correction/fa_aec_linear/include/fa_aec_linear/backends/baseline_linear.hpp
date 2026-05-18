#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fa_aec_linear::backends
{

struct BaselineLinearConfig
{
  int channels{-1};
  std::string encoding{};
  int bit_depth{-1};
  double cancel_gain{0.0};
};

enum class ProcessStatus
{
  kOk,
  kEmptyMic,
  kEmptyReference,
  kMisalignedMic,
  kMisalignedReference,
  kUnsupportedFormat,
  kSampleCountMismatch,
  kNonFiniteMic,
  kNonFiniteReference,
  kOutOfRangeMic,
  kOutOfRangeReference,
  kNonFiniteOutput,
  kOutOfRangeOutput,
  kPcm16OutputOutOfRange,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
};

class BaselineLinearBackend
{
public:
  static constexpr const char * kName = "baseline_linear";

  explicit BaselineLinearBackend(const BaselineLinearConfig & config);

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & mic_input,
    const std::vector<uint8_t> & reference_input,
    std::vector<uint8_t> & output) const;

private:
  [[nodiscard]] ProcessStatus decodeToFloat(
    const std::vector<uint8_t> & input,
    bool is_reference,
    std::vector<float> & samples) const;
  [[nodiscard]] ProcessStatus encodeFromFloat(
    const std::vector<float> & samples,
    std::vector<uint8_t> & output) const;
  [[nodiscard]] size_t bytesPerFrame() const;
  [[nodiscard]] size_t bytesPerSample() const;

  BaselineLinearConfig config_;
};

bool isSupportedAudioFormatPair(const std::string & encoding, int bit_depth);
const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_aec_linear::backends

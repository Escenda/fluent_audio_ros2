#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_mix::backends
{

struct InternalPcm16MixerConfig
{
  int channels{0};
  std::vector<double> input_gains_db{};
};

enum class MixStatus
{
  kOk,
  kNoInputs,
  kInputCountMismatch,
  kEmptyInput,
  kMisalignedInput,
  kSampleCountMismatch,
  kNonFiniteGain,
  kNonFiniteOutput,
  kOutOfRangeOutput,
  kPcm16RangeOutput,
};

struct MixResult
{
  MixStatus status{MixStatus::kOk};
  size_t input_count{0U};
  size_t sample_count{0U};
};

class InternalPcm16MixerBackend
{
public:
  static constexpr const char * kName = "internal_pcm16_mixer";

  explicit InternalPcm16MixerBackend(const InternalPcm16MixerConfig & config);

  [[nodiscard]] int channels() const;
  [[nodiscard]] size_t inputCount() const;
  [[nodiscard]] const std::vector<double> & inputGainsDb() const;
  [[nodiscard]] const std::vector<double> & inputGainsLinear() const;
  [[nodiscard]] size_t lastSampleCount() const;

  [[nodiscard]] MixResult mix(
    const std::vector<std::vector<uint8_t>> & inputs,
    std::vector<uint8_t> & output);

private:
  [[nodiscard]] MixStatus decodePcm16Le(
    const std::vector<uint8_t> & input,
    std::vector<float> & samples,
    size_t & frame_count) const;
  [[nodiscard]] MixStatus encodePcm16Le(
    const std::vector<float> & samples,
    std::vector<uint8_t> & output) const;

  InternalPcm16MixerConfig config_;
  std::vector<double> input_gains_linear_{};
  size_t last_sample_count_{0U};
};

double dbToLinear(double db);
const char * mixStatusMessage(MixStatus status);

}  // namespace fa_mix::backends

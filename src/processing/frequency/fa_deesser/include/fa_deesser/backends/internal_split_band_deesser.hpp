#pragma once

#include <cstdint>
#include <vector>

namespace fa_deesser::backends
{

struct InternalSplitBandDeesserConfig
{
  int sample_rate{-1};
  int channels{-1};
  double cutoff_hz{-1.0};
  double threshold{-1.0};
  double attenuation_db{1.0};
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kNonFiniteInput,
  kOutOfRangeInput,
  kNonFiniteOutput,
  kOutOfRangeOutput,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  uint64_t samples_attenuated{0};
};

class InternalSplitBandDeesserBackend
{
public:
  static constexpr const char * kName = "internal_split_band_deesser";

  explicit InternalSplitBandDeesserBackend(const InternalSplitBandDeesserConfig & config);

  [[nodiscard]] double alpha() const;
  [[nodiscard]] double attenuationGain() const;
  [[nodiscard]] int channels() const;

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output,
    bool reset_state);

private:
  InternalSplitBandDeesserConfig config_;
  double alpha_{0.0};
  double attenuation_gain_{1.0};
  std::vector<double> low_band_state_{};
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_deesser::backends

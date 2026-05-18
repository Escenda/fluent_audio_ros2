#pragma once

#include <cstdint>
#include <vector>

namespace fa_eq::backends
{

struct InternalThreeBandEqConfig
{
  int sample_rate{-1};
  int channels{-1};
  double low_cutoff_hz{-1.0};
  double high_cutoff_hz{-1.0};
  double gain_low_db{0.0};
  double gain_mid_db{0.0};
  double gain_high_db{0.0};
};

struct ChannelFilterState
{
  float previous_low_output{0.0F};
  float previous_hp_input{0.0F};
  float previous_hp_output{0.0F};
  bool initialized{false};
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

class InternalThreeBandEqBackend
{
public:
  static constexpr const char * kName = "internal_three_band_eq";

  explicit InternalThreeBandEqBackend(const InternalThreeBandEqConfig & config);

  [[nodiscard]] double lowAlpha() const;
  [[nodiscard]] double highAlpha() const;
  [[nodiscard]] double gainLowLinear() const;
  [[nodiscard]] double gainMidLinear() const;
  [[nodiscard]] double gainHighLinear() const;
  [[nodiscard]] int channels() const;

  [[nodiscard]] ProcessStatus process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output,
    bool reset_state);

private:
  InternalThreeBandEqConfig config_;
  double low_alpha_{0.0};
  double high_alpha_{0.0};
  double gain_low_linear_{1.0};
  double gain_mid_linear_{1.0};
  double gain_high_linear_{1.0};
  std::vector<ChannelFilterState> channel_states_{};
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_eq::backends

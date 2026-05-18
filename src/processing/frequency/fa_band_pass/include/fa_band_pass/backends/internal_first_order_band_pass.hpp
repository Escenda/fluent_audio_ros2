#pragma once

#include <cstdint>
#include <vector>

namespace fa_band_pass::backends
{

struct InternalFirstOrderBandPassConfig
{
  int sample_rate{-1};
  int channels{-1};
  double low_cut_hz{-1.0};
  double high_cut_hz{-1.0};
};

struct ChannelFilterState
{
  float previous_hp_input{0.0F};
  float previous_hp_output{0.0F};
  float previous_lp_output{0.0F};
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

class InternalFirstOrderBandPassBackend
{
public:
  static constexpr const char * kName = "internal_first_order_band_pass";

  explicit InternalFirstOrderBandPassBackend(const InternalFirstOrderBandPassConfig & config);

  [[nodiscard]] double highPassAlpha() const;
  [[nodiscard]] double lowPassAlpha() const;
  [[nodiscard]] int channels() const;
  [[nodiscard]] double lowCutHz() const;
  [[nodiscard]] double highCutHz() const;

  [[nodiscard]] ProcessStatus process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output,
    bool reset_state);

private:
  InternalFirstOrderBandPassConfig config_;
  double hp_alpha_{0.0};
  double lp_alpha_{0.0};
  std::vector<ChannelFilterState> channel_states_{};
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_band_pass::backends

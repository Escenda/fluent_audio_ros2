#pragma once

#include <cstdint>
#include <vector>

namespace fa_low_pass::backends
{

struct InternalFirstOrderLowPassConfig
{
  int sample_rate{-1};
  int channels{-1};
  double cutoff_hz{-1.0};
};

struct ChannelFilterState
{
  float previous_output{0.0F};
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

class InternalFirstOrderLowPassBackend
{
public:
  static constexpr const char * kName = "internal_first_order_low_pass";

  explicit InternalFirstOrderLowPassBackend(const InternalFirstOrderLowPassConfig & config);

  [[nodiscard]] double alpha() const;
  [[nodiscard]] int channels() const;
  [[nodiscard]] double cutoffHz() const;

  [[nodiscard]] ProcessStatus process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output,
    bool reset_state);

private:
  InternalFirstOrderLowPassConfig config_;
  double alpha_{0.0};
  std::vector<ChannelFilterState> channel_states_{};
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_low_pass::backends

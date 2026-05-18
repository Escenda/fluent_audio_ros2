#pragma once

#include <cstdint>
#include <vector>

namespace fa_notch::backends
{

struct InternalNotchConfig
{
  int sample_rate{-1};
  int channels{-1};
  double center_hz{-1.0};
  double q{-1.0};
};

struct BiquadCoefficients
{
  double b0{0.0};
  double b1{0.0};
  double b2{0.0};
  double a1{0.0};
  double a2{0.0};
};

struct ChannelFilterState
{
  double previous_input_1{0.0};
  double previous_input_2{0.0};
  double previous_output_1{0.0};
  double previous_output_2{0.0};
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kNonFiniteInput,
  kNonFiniteOutput,
};

class InternalNotchBackend
{
public:
  static constexpr const char * kName = "internal_notch";

  explicit InternalNotchBackend(const InternalNotchConfig & config);

  [[nodiscard]] double centerHz() const;
  [[nodiscard]] double q() const;
  [[nodiscard]] int channels() const;
  [[nodiscard]] const BiquadCoefficients & coefficients() const;

  [[nodiscard]] ProcessStatus process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output);

private:
  InternalNotchConfig config_;
  BiquadCoefficients coefficients_{};
  std::vector<ChannelFilterState> channel_states_{};
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_notch::backends

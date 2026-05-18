#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_hum::backends
{

struct InternalNotchCascadeConfig
{
  int sample_rate{-1};
  int channels{-1};
  double frequency_hz{-1.0};
  int harmonics{-1};
  double q{-1.0};
};

enum class ProcessStatus
{
  kOk,
  kEmptySourceId,
  kEmptyInput,
  kMisalignedInput,
  kNonFiniteInput,
  kOutOfRangeInput,
  kStaleEpoch,
  kNonFiniteOutput,
  kOutOfRangeOutput,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  bool source_reset{false};
  bool epoch_reset{false};
};

class InternalNotchCascadeBackend
{
public:
  static constexpr const char * kName = "internal_notch_cascade";

  explicit InternalNotchCascadeBackend(const InternalNotchCascadeConfig & config);

  [[nodiscard]] ProcessResult process(
    const std::string & source_id,
    uint32_t epoch,
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output);

  [[nodiscard]] size_t stageCount() const;
  [[nodiscard]] std::vector<double> centerFrequenciesHz() const;
  [[nodiscard]] const std::string & activeSourceId() const;
  [[nodiscard]] bool hasActiveStream() const;
  [[nodiscard]] uint32_t activeEpoch() const;

private:
  struct BiquadCoefficients
  {
    double center_hz{0.0};
    double b0{0.0};
    double b1{0.0};
    double b2{0.0};
    double a1{0.0};
    double a2{0.0};
  };

  struct BiquadState
  {
    double previous_input_1{0.0};
    double previous_input_2{0.0};
    double previous_output_1{0.0};
    double previous_output_2{0.0};
  };

  using ChannelCascadeState = std::vector<BiquadState>;

  [[nodiscard]] std::vector<ChannelCascadeState> zeroedChannelStates() const;

  InternalNotchCascadeConfig config_;
  std::vector<BiquadCoefficients> cascade_coefficients_{};
  bool has_active_stream_{false};
  std::string active_source_id_{};
  uint32_t active_epoch_{0};
  std::vector<ChannelCascadeState> channel_states_{};
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_hum::backends

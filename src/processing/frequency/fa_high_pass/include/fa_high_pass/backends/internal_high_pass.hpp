#pragma once

#include <cstdint>
#include <vector>

namespace fa_high_pass::backends
{

struct InternalHighPassConfig
{
  int sample_rate{-1};
  int channels{-1};
  double cutoff_hz{-1.0};
};

struct ChannelFilterState
{
  float previous_input{0.0F};
  float previous_output{0.0F};
  bool initialized{false};
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kNonFiniteInput,
  kNonFiniteOutput,
};

class InternalHighPassBackend
{
public:
  static constexpr const char * kName = "internal_high_pass";

  explicit InternalHighPassBackend(const InternalHighPassConfig & config);

  [[nodiscard]] double alpha() const;
  [[nodiscard]] int channels() const;
  [[nodiscard]] double cutoffHz() const;

  [[nodiscard]] ProcessStatus process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output);

private:
  InternalHighPassConfig config_;
  double alpha_{0.0};
  std::vector<ChannelFilterState> channel_states_{};
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_high_pass::backends

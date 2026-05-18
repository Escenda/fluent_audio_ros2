#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace fa_delay::backends
{

struct InternalSampleDelayConfig
{
  int channels{0};
  size_t delay_samples{0U};
};

enum class ProcessStatus
{
  kOk,
  kEmptySourceId,
  kEmptyInput,
  kMisalignedInput,
  kInvalidState,
  kNonFiniteInput,
  kOutOfRangeInput,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  bool source_reset{false};
};

class InternalSampleDelayBackend
{
public:
  static constexpr const char * kName = "internal_sample_delay";

  explicit InternalSampleDelayBackend(const InternalSampleDelayConfig & config);

  [[nodiscard]] int channels() const;
  [[nodiscard]] size_t delaySamples() const;
  [[nodiscard]] const std::string & currentSourceId() const;

  [[nodiscard]] ProcessResult process(
    const std::string & source_id,
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output);

private:
  void resetDelayState(std::vector<std::deque<float>> & buffers) const;

  [[nodiscard]] bool validateDelayState(
    const std::vector<std::deque<float>> & buffers) const;

  InternalSampleDelayConfig config_;
  std::string current_source_id_{};
  std::vector<std::deque<float>> delay_buffers_{};
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_delay::backends

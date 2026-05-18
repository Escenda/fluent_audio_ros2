#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_echo::backends
{

struct InternalFeedbackEchoConfig
{
  int channels{-1};
  size_t delay_samples{0};
  double feedback_gain{0.0};
  double wet_gain{0.0};
  double dry_gain{0.0};
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
  kNonFiniteOutput,
  kOutOfRangeOutput,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  bool source_reset{false};
};

class InternalFeedbackEchoBackend
{
public:
  static constexpr const char * kName = "internal_feedback_echo";

  explicit InternalFeedbackEchoBackend(const InternalFeedbackEchoConfig & config);

  [[nodiscard]] int channels() const;
  [[nodiscard]] size_t delaySamples() const;
  [[nodiscard]] double feedbackGain() const;
  [[nodiscard]] double wetGain() const;
  [[nodiscard]] double dryGain() const;
  [[nodiscard]] const std::string & currentSourceId() const;

  [[nodiscard]] ProcessResult process(
    const std::string & source_id,
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output);

private:
  void resetDelayState(
    std::vector<std::vector<float>> & buffers,
    std::vector<size_t> & positions) const;

  InternalFeedbackEchoConfig config_;
  std::string current_source_id_{};
  std::vector<std::vector<float>> delay_buffers_{};
  std::vector<size_t> delay_positions_{};
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_echo::backends

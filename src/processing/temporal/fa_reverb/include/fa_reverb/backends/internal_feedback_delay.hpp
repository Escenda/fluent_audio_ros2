#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_reverb::backends
{

struct InternalFeedbackDelayConfig
{
  int sample_rate{0};
  int channels{0};
  double room_size{-1.0};
  double damping{-1.0};
  double wet_gain{-1.0};
  double dry_gain{-1.0};
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

struct DelayLineState
{
  std::vector<float> buffer{};
  size_t position{0U};
  float filter_state{0.0F};
};

class InternalFeedbackDelayBackend
{
public:
  static constexpr const char * kName = "internal_feedback_delay";

  explicit InternalFeedbackDelayBackend(const InternalFeedbackDelayConfig & config);

  [[nodiscard]] int sampleRate() const;
  [[nodiscard]] int channels() const;
  [[nodiscard]] double roomSize() const;
  [[nodiscard]] double damping() const;
  [[nodiscard]] double wetGain() const;
  [[nodiscard]] double dryGain() const;
  [[nodiscard]] double effectiveFeedbackGain() const;
  [[nodiscard]] size_t delayLineCount() const;
  [[nodiscard]] const std::string & currentSourceId() const;

  [[nodiscard]] ProcessResult process(
    const std::string & source_id,
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output);

private:
  void resetReverbState(std::vector<std::vector<DelayLineState>> & state) const;
  [[nodiscard]] bool validateReverbState(
    const std::vector<std::vector<DelayLineState>> & state) const;

  InternalFeedbackDelayConfig config_;
  std::vector<size_t> delay_samples_{};
  double effective_feedback_gain_{0.0};
  std::string current_source_id_{};
  std::vector<std::vector<DelayLineState>> delay_lines_{};
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_reverb::backends

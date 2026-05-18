#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_agc::backends
{

struct InternalRmsAgcConfig
{
  InternalRmsAgcConfig() = delete;
  InternalRmsAgcConfig(
    int sample_rate_value,
    int channels_value,
    double target_rms_value,
    double min_gain_value,
    double max_gain_value,
    double attack_ms_value,
    double release_ms_value);

  int sample_rate;
  int channels;
  double target_rms;
  double min_gain;
  double max_gain;
  double attack_ms;
  double release_ms;
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kNonFiniteInput,
  kOutOfRangeInput,
  kNonFiniteGain,
  kNonFiniteOutput,
  kOutOfRangeOutput,
};

enum class GainDirection
{
  kUnchanged,
  kReduction,
  kIncrease,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  double frame_rms{0.0};
  double target_gain{1.0};
  double committed_gain{1.0};
  GainDirection gain_direction{GainDirection::kUnchanged};
};

class InternalRmsAgcBackend
{
public:
  static constexpr const char * kName = "internal_rms_agc";

  explicit InternalRmsAgcBackend(const InternalRmsAgcConfig & config);

  [[nodiscard]] double targetRms() const;
  [[nodiscard]] double minGain() const;
  [[nodiscard]] double maxGain() const;
  [[nodiscard]] double attackMs() const;
  [[nodiscard]] double releaseMs() const;
  [[nodiscard]] double currentGain() const;
  [[nodiscard]] double lastFrameRms() const;
  [[nodiscard]] double lastTargetGain() const;

  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output);

private:
  [[nodiscard]] double calculateFrameRms(const std::vector<float> & samples) const;
  [[nodiscard]] double boundedTargetGain(double frame_rms) const;
  [[nodiscard]] double smoothingAlpha(double time_constant_ms, size_t sample_count) const;
  [[nodiscard]] double smoothedGain(double target_gain, size_t sample_count) const;

  InternalRmsAgcConfig config_;
  double current_gain_{1.0};
  double last_frame_rms_{0.0};
  double last_target_gain_{1.0};
};

const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_agc::backends

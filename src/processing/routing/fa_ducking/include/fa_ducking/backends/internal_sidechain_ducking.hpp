#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_ducking::backends
{

struct InternalSidechainDuckingConfig
{
  int channels{0};
  int sample_rate{0};
  double sidechain_threshold_rms{-1.0};
  int64_t sidechain_max_age_ns{-1};
  double ducking_gain_db{0.0};
  double attack_ms{-1.0};
  double release_ms{-1.0};
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kNonFiniteInput,
  kOutOfRangeInput,
  kInvalidRms,
  kInvalidGain,
  kNonFiniteOutput,
  kOutOfRangeOutput,
};

struct SidechainResult
{
  ProcessStatus status{ProcessStatus::kOk};
  double rms{0.0};
  size_t frame_count{0U};
};

struct ProgramResult
{
  ProcessStatus status{ProcessStatus::kOk};
  bool sidechain_active{false};
  bool sidechain_stale{false};
  double sidechain_rms{0.0};
  int64_t sidechain_age_ns{-1};
  double target_gain{1.0};
  double output_gain{1.0};
  size_t frame_count{0U};
};

class InternalSidechainDuckingBackend
{
public:
  static constexpr const char * kName = "internal_sidechain_ducking";

  explicit InternalSidechainDuckingBackend(const InternalSidechainDuckingConfig & config);

  [[nodiscard]] int channels() const;
  [[nodiscard]] int sampleRate() const;
  [[nodiscard]] double sidechainThresholdRms() const;
  [[nodiscard]] int64_t sidechainMaxAgeNs() const;
  [[nodiscard]] double duckingGainDb() const;
  [[nodiscard]] double duckingGainLinear() const;
  [[nodiscard]] double attackMs() const;
  [[nodiscard]] double releaseMs() const;
  [[nodiscard]] double currentGain() const;
  [[nodiscard]] double lastTargetGain() const;
  [[nodiscard]] double lastSidechainRms() const;
  [[nodiscard]] int64_t lastSidechainAgeNs() const;
  [[nodiscard]] bool lastSidechainActive() const;
  [[nodiscard]] bool hasSidechain() const;

  [[nodiscard]] SidechainResult observeSidechain(const std::vector<uint8_t> & input, int64_t now_ns);
  void invalidateSidechain();
  [[nodiscard]] ProgramResult processProgram(
    const std::vector<uint8_t> & input,
    int64_t now_ns,
    std::vector<uint8_t> & output);

private:
  [[nodiscard]] ProcessStatus validateAndMeasure(
    const std::vector<uint8_t> & input,
    double & rms,
    size_t & frame_count) const;
  [[nodiscard]] double smoothingAlpha(double time_constant_ms, size_t frame_count) const;
  [[nodiscard]] double smoothedGain(double target_gain, size_t frame_count) const;

  InternalSidechainDuckingConfig config_;
  double ducking_gain_linear_{1.0};
  bool has_sidechain_{false};
  double latest_sidechain_rms_{0.0};
  int64_t latest_sidechain_received_ns_{0};
  double current_gain_{1.0};
  double last_target_gain_{1.0};
  double last_sidechain_rms_{0.0};
  int64_t last_sidechain_age_ns_{-1};
  bool last_sidechain_active_{false};
};

double dbToLinear(double db);
const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_ducking::backends

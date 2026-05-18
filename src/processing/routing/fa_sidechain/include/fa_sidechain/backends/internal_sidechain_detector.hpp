#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_sidechain::backends
{

struct InternalSidechainDetectorConfig
{
  int channels{0};
  double threshold_rms{-1.0};
  double active_gain_db{0.0};
  double inactive_gain_db{0.0};
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

struct DetectionResult
{
  ProcessStatus status{ProcessStatus::kOk};
  double rms{0.0};
  double gain_linear{1.0};
  bool active{false};
  size_t frame_count{0U};
};

class InternalSidechainDetectorBackend
{
public:
  static constexpr const char * kName = "internal_sidechain_detector";

  explicit InternalSidechainDetectorBackend(const InternalSidechainDetectorConfig & config);

  [[nodiscard]] int channels() const;
  [[nodiscard]] double thresholdRms() const;
  [[nodiscard]] double activeGainDb() const;
  [[nodiscard]] double inactiveGainDb() const;
  [[nodiscard]] double activeGainLinear() const;
  [[nodiscard]] double inactiveGainLinear() const;
  [[nodiscard]] double lastRms() const;
  [[nodiscard]] double lastGainLinear() const;
  [[nodiscard]] bool lastActive() const;

  [[nodiscard]] DetectionResult detect(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & control_data);

private:
  [[nodiscard]] ProcessStatus validateAndMeasure(
    const std::vector<uint8_t> & input,
    double & rms,
    size_t & frame_count) const;

  InternalSidechainDetectorConfig config_;
  double active_gain_linear_{1.0};
  double inactive_gain_linear_{1.0};
  double last_rms_{0.0};
  double last_gain_linear_{1.0};
  bool last_active_{false};
};

double dbToLinear(double db);
const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_sidechain::backends

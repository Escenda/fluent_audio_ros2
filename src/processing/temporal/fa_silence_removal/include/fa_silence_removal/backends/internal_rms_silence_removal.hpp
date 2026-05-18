#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_silence_removal::backends
{

struct InternalRmsSilenceRemovalConfig
{
  int channels{0};
  double threshold_rms{-1.0};
  size_t hangover_samples{0U};
};

enum class ProcessStatus
{
  kOk,
  kEmptyInput,
  kMisalignedInput,
  kNonFiniteInput,
  kOutOfRangeInput,
  kInvalidRms,
};

enum class Decision
{
  kAcceptedActive,
  kAcceptedHangover,
  kDroppedSilent,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  Decision decision{Decision::kDroppedSilent};
  double rms{0.0};
  size_t frame_count{0U};
  size_t hangover_samples_remaining{0U};
};

class InternalRmsSilenceRemovalBackend
{
public:
  static constexpr const char * kName = "internal_rms_silence_removal";

  explicit InternalRmsSilenceRemovalBackend(const InternalRmsSilenceRemovalConfig & config);

  [[nodiscard]] int channels() const;
  [[nodiscard]] double thresholdRms() const;
  [[nodiscard]] size_t hangoverSamples() const;
  [[nodiscard]] size_t hangoverSamplesRemaining() const;
  [[nodiscard]] double lastRms() const;

  [[nodiscard]] ProcessResult process(const std::vector<uint8_t> & input);

private:
  void consumeHangoverSamples(size_t frame_count);

  InternalRmsSilenceRemovalConfig config_;
  size_t hangover_samples_remaining_{0U};
  double last_rms_{0.0};
};

const char * decisionName(Decision decision);
const char * processStatusMessage(ProcessStatus status);

}  // namespace fa_silence_removal::backends

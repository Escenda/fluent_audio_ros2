#pragma once

#include <cstdint>
#include <deque>
#include <vector>

#include "fa_audio_window/audio_format.hpp"
#include "fa_audio_window/time_range.hpp"

namespace fa_audio_window
{

struct TimedAudioFrame
{
  TimeRange time_range{};
  uint32_t epoch{0};
  std::vector<uint8_t> data{};
};

enum class WindowQueryStatus
{
  kOk,
  kWindowNotFound,
  kRangeOutsideWindow,
  kRangeNotContinuous,
  kNoSamplesSelected,
};

struct WindowQueryResult
{
  WindowQueryStatus status{WindowQueryStatus::kWindowNotFound};
  TimeRange exported_range{};
  std::vector<uint8_t> pcm_data{};
};

class AudioWindowBuffer
{
public:
  AudioWindowBuffer(uint64_t retention_ns, AudioFormat format);

  void addFrame(const TimedAudioFrame & frame);
  WindowQueryResult query(const TimeRange & range) const;
  bool empty() const;
  TimeRange retainedRange() const;
  uint64_t retentionNs() const;

private:
  void pruneExpiredFrames();
  size_t sampleCount(const TimedAudioFrame & frame) const;
  int64_t frameTimeForSample(const TimedAudioFrame & frame, size_t sample_index) const;
  size_t firstSampleIndexAtOrAfter(const TimedAudioFrame & frame, int64_t unix_ns) const;
  size_t firstSampleIndexAtOrAfterEnd(const TimedAudioFrame & frame, int64_t unix_ns) const;

  uint64_t retention_ns_{0};
  AudioFormat format_{};
  std::deque<TimedAudioFrame> frames_{};
  int64_t newest_end_unix_ns_{0};
};

}  // namespace fa_audio_window

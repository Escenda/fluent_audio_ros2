#include "fa_audio_window/audio_window_buffer.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fa_audio_window
{

namespace
{
constexpr int64_t kNsecPerSecond = 1000000000LL;

int64_t checkedSampleTime(const int64_t start_unix_ns, const size_t sample_index, const uint32_t sample_rate)
{
  const size_t whole_seconds = sample_index / sample_rate;
  const size_t remaining_samples = sample_index % sample_rate;
  if (whole_seconds > static_cast<size_t>(std::numeric_limits<int64_t>::max() / kNsecPerSecond)) {
    throw std::runtime_error("sample timestamp overflow");
  }
  const int64_t whole_ns = static_cast<int64_t>(whole_seconds) * kNsecPerSecond;
  const int64_t remaining_ns =
    static_cast<int64_t>(remaining_samples * static_cast<size_t>(kNsecPerSecond) / sample_rate);
  if (start_unix_ns > std::numeric_limits<int64_t>::max() - whole_ns - remaining_ns) {
    throw std::runtime_error("sample timestamp overflow");
  }
  return start_unix_ns + whole_ns + remaining_ns;
}

size_t floorSampleIndex(const int64_t delta_ns, const uint32_t sample_rate)
{
  const size_t whole_seconds = static_cast<size_t>(delta_ns / kNsecPerSecond);
  const size_t remaining_ns = static_cast<size_t>(delta_ns % kNsecPerSecond);
  return whole_seconds * static_cast<size_t>(sample_rate) +
    remaining_ns * static_cast<size_t>(sample_rate) / static_cast<size_t>(kNsecPerSecond);
}

size_t ceilSampleIndex(const int64_t delta_ns, const uint32_t sample_rate)
{
  const size_t floor_index = floorSampleIndex(delta_ns, sample_rate);
  const int64_t floor_time_ns = checkedSampleTime(0, floor_index, sample_rate);
  if (floor_time_ns == delta_ns) {
    return floor_index;
  }
  return floor_index + 1u;
}
}  // namespace

AudioWindowBuffer::AudioWindowBuffer(const uint64_t retention_ns, AudioFormat format)
: retention_ns_(retention_ns), format_(std::move(format))
{
  if (retention_ns_ == 0u) {
    throw std::runtime_error("retention_ns must be > 0");
  }
  if (format_.sample_rate == 0u) {
    throw std::runtime_error("sample_rate must be > 0");
  }
  (void)bytesPerSampleFrame(format_);
}

void AudioWindowBuffer::addFrame(const TimedAudioFrame & frame)
{
  if (!isValidRange(frame.time_range)) {
    throw std::runtime_error("TimedAudioFrame time_range must be valid");
  }
  if (frame.data.empty()) {
    throw std::runtime_error("TimedAudioFrame data is required");
  }
  if (frame.data.size() % bytesPerSampleFrame(format_) != 0u) {
    throw std::runtime_error("TimedAudioFrame data is not aligned to sample frames");
  }

  const auto insert_at = std::lower_bound(
    frames_.begin(),
    frames_.end(),
    frame.time_range.start_unix_ns,
    [](const TimedAudioFrame & candidate, const int64_t start_unix_ns) {
      return candidate.time_range.start_unix_ns < start_unix_ns;
    });
  if (insert_at != frames_.begin()) {
    const auto previous = std::prev(insert_at);
    if (previous->time_range.end_unix_ns > frame.time_range.start_unix_ns) {
      throw std::runtime_error("TimedAudioFrame overlaps a retained frame");
    }
  }
  if (insert_at != frames_.end() && frame.time_range.end_unix_ns > insert_at->time_range.start_unix_ns) {
    throw std::runtime_error("TimedAudioFrame overlaps a retained frame");
  }
  frames_.insert(insert_at, frame);
  newest_end_unix_ns_ = std::max(newest_end_unix_ns_, frame.time_range.end_unix_ns);
  pruneExpiredFrames();
}

WindowQueryResult AudioWindowBuffer::query(const TimeRange & range) const
{
  if (frames_.empty()) {
    return {WindowQueryStatus::kWindowNotFound, {}, {}};
  }
  if (!isValidRange(range)) {
    return {WindowQueryStatus::kRangeOutsideWindow, {}, {}};
  }

  const TimeRange retained = retainedRange();
  if (range.start_unix_ns < retained.start_unix_ns || range.end_unix_ns > retained.end_unix_ns) {
    return {WindowQueryStatus::kRangeOutsideWindow, {}, {}};
  }

  WindowQueryResult result;
  result.status = WindowQueryStatus::kOk;
  bool has_selected_samples = false;
  int64_t coverage_cursor = range.start_unix_ns;
  const size_t sample_frame_bytes = bytesPerSampleFrame(format_);

  for (const TimedAudioFrame & frame : frames_) {
    if (frame.time_range.end_unix_ns <= coverage_cursor) {
      continue;
    }
    if (frame.time_range.start_unix_ns >= range.end_unix_ns) {
      break;
    }
    if (frame.time_range.start_unix_ns > coverage_cursor) {
      return {WindowQueryStatus::kRangeNotContinuous, {}, {}};
    }

    const size_t first_index = firstSampleIndexAtOrAfter(frame, range.start_unix_ns);
    const size_t end_index = firstSampleIndexAtOrAfterEnd(frame, range.end_unix_ns);
    if (end_index <= first_index) {
      continue;
    }

    const size_t first_byte = first_index * sample_frame_bytes;
    const size_t end_byte = end_index * sample_frame_bytes;
    result.pcm_data.insert(
      result.pcm_data.end(),
      frame.data.begin() + static_cast<std::ptrdiff_t>(first_byte),
      frame.data.begin() + static_cast<std::ptrdiff_t>(end_byte));

    const int64_t selected_start = frameTimeForSample(frame, first_index);
    const int64_t selected_end = frameTimeForSample(frame, end_index);
    if (!has_selected_samples) {
      result.exported_range = {selected_start, selected_end};
      has_selected_samples = true;
    } else {
      if (selected_start > result.exported_range.end_unix_ns) {
        return {WindowQueryStatus::kRangeNotContinuous, {}, {}};
      }
      result.exported_range.end_unix_ns = selected_end;
    }
    coverage_cursor = std::max(coverage_cursor, std::min(frame.time_range.end_unix_ns, range.end_unix_ns));
  }

  if (!has_selected_samples) {
    return {WindowQueryStatus::kNoSamplesSelected, {}, {}};
  }
  if (coverage_cursor < range.end_unix_ns) {
    return {WindowQueryStatus::kRangeNotContinuous, {}, {}};
  }
  return result;
}

bool AudioWindowBuffer::empty() const
{
  return frames_.empty();
}

TimeRange AudioWindowBuffer::retainedRange() const
{
  if (frames_.empty()) {
    return {};
  }
  return {frames_.front().time_range.start_unix_ns, newest_end_unix_ns_};
}

uint64_t AudioWindowBuffer::retentionNs() const
{
  return retention_ns_;
}

void AudioWindowBuffer::pruneExpiredFrames()
{
  const int64_t retention_start = newest_end_unix_ns_ > static_cast<int64_t>(retention_ns_) ?
    newest_end_unix_ns_ - static_cast<int64_t>(retention_ns_) :
    0;
  while (!frames_.empty() && frames_.front().time_range.end_unix_ns <= retention_start) {
    frames_.pop_front();
  }
}

size_t AudioWindowBuffer::sampleCount(const TimedAudioFrame & frame) const
{
  return frame.data.size() / bytesPerSampleFrame(format_);
}

int64_t AudioWindowBuffer::frameTimeForSample(const TimedAudioFrame & frame, const size_t sample_index) const
{
  return checkedSampleTime(frame.time_range.start_unix_ns, sample_index, format_.sample_rate);
}

size_t AudioWindowBuffer::firstSampleIndexAtOrAfter(const TimedAudioFrame & frame, const int64_t unix_ns) const
{
  if (unix_ns <= frame.time_range.start_unix_ns) {
    return 0u;
  }
  const size_t index = ceilSampleIndex(unix_ns - frame.time_range.start_unix_ns, format_.sample_rate);
  return std::min(index, sampleCount(frame));
}

size_t AudioWindowBuffer::firstSampleIndexAtOrAfterEnd(const TimedAudioFrame & frame, const int64_t unix_ns) const
{
  if (unix_ns >= frame.time_range.end_unix_ns) {
    return sampleCount(frame);
  }
  if (unix_ns <= frame.time_range.start_unix_ns) {
    return 0u;
  }
  const size_t index = ceilSampleIndex(unix_ns - frame.time_range.start_unix_ns, format_.sample_rate);
  return std::min(index, sampleCount(frame));
}

}  // namespace fa_audio_window

#pragma once

#include <cstdint>

namespace fa_audio_window
{

struct TimeRange
{
  int64_t start_unix_ns{0};
  int64_t end_unix_ns{0};
};

inline bool isValidRange(const TimeRange & range)
{
  return range.start_unix_ns >= 0 && range.end_unix_ns > range.start_unix_ns;
}

}  // namespace fa_audio_window

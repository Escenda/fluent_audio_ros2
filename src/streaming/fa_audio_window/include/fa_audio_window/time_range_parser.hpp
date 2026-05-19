#pragma once

#include <string>

#include "fa_audio_window/time_range.hpp"

namespace fa_audio_window
{

struct TimeRangeParseResult
{
  bool success{false};
  TimeRange range{};
  std::string message{};
};

TimeRangeParseResult parseNumericUnixNsRange(const std::string & spec);

}  // namespace fa_audio_window

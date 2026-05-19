#include "fa_audio_window/time_range_parser.hpp"

#include <cctype>
#include <charconv>
#include <string>
#include <string_view>

namespace fa_audio_window
{

namespace
{
std::string_view trim(std::string_view value)
{
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

bool parseInt64(std::string_view text, int64_t & value)
{
  if (text.empty()) {
    return false;
  }
  const char * begin = text.data();
  const char * end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  return result.ec == std::errc{} && result.ptr == end;
}
}  // namespace

TimeRangeParseResult parseNumericUnixNsRange(const std::string & spec)
{
  const std::string_view trimmed = trim(spec);
  const size_t separator = trimmed.find("..");
  if (separator == std::string_view::npos || trimmed.find("..", separator + 2u) != std::string_view::npos) {
    return {false, {}, "time_range_spec must be numeric start_unix_ns..end_unix_ns"};
  }

  int64_t start = 0;
  int64_t end = 0;
  if (!parseInt64(trim(trimmed.substr(0u, separator)), start) ||
      !parseInt64(trim(trimmed.substr(separator + 2u)), end))
  {
    return {false, {}, "only numeric unix nanosecond ranges are supported"};
  }

  const TimeRange range{start, end};
  if (!isValidRange(range)) {
    return {false, {}, "time_range_spec must resolve to 0 <= start < end"};
  }
  return {true, range, ""};
}

}  // namespace fa_audio_window

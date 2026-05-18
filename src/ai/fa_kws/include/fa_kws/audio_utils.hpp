#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_kws
{

std::vector<float> frameToCanonicalFloat(
  const fa_interfaces::msg::AudioFrame &msg,
  const std::string &expected_source_id,
  const std::string &expected_stream_id);

}  // namespace fa_kws

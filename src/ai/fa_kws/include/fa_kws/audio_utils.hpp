#pragma once

#include <cstdint>
#include <vector>

#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_kws
{

std::vector<float> frameToCanonicalFloat(const fa_interfaces::msg::AudioFrame &msg);

}  // namespace fa_kws

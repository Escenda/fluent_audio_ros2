#pragma once

#include <cstdint>
#include <vector>

#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_kws
{

std::vector<float> frameToMonoFloat(const fa_interfaces::msg::AudioFrame &msg);

std::vector<float> resampleLinear(const std::vector<float> &samples,
                                  std::int32_t src_rate,
                                  std::int32_t dst_rate);

}  // namespace fa_kws


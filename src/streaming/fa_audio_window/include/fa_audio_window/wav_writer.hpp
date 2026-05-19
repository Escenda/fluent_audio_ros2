#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "fa_audio_window/audio_format.hpp"

namespace fa_audio_window
{

class WavWriter
{
public:
  static void writePcm16Le(
    const std::filesystem::path & path,
    const AudioFormat & format,
    const std::vector<uint8_t> & pcm_data);
};

}  // namespace fa_audio_window

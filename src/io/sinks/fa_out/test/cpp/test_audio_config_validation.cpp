#include "fa_out/audio_config_validation.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace
{

TEST(FaOutAudioConfigValidation, RejectsNonPositiveAndOutOfRangeUnsignedParameters)
{
  EXPECT_EQ(fa_out::validation::requirePositiveUint32("audio.sample_rate", 48000), 48000u);
  EXPECT_EQ(fa_out::validation::requirePositiveSize("queue.max_frames", 32), 32u);
  EXPECT_THROW(
    fa_out::validation::requirePositiveUint32("audio.sample_rate", 0),
    std::invalid_argument);
  EXPECT_THROW(
    fa_out::validation::requirePositiveSize("queue.max_frames", -1),
    std::invalid_argument);
  EXPECT_THROW(
    fa_out::validation::requirePositiveUint32(
      "audio.sample_rate",
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1),
    std::invalid_argument);
}

TEST(FaOutAudioConfigValidation, RejectsNonIntegerPlaybackChunkFrameCounts)
{
  EXPECT_EQ(fa_out::validation::playbackChunkFrames(48000, 30), 1440u);
  EXPECT_THROW(
    fa_out::validation::playbackChunkFrames(44100, 1),
    std::invalid_argument);
}

TEST(FaOutAudioConfigValidation, RejectsByteCountOverflow)
{
  EXPECT_EQ(fa_out::validation::bytesPerFrame(2, 16), 4u);
  EXPECT_THROW(fa_out::validation::bytesPerFrame(1, 7), std::invalid_argument);
  EXPECT_THROW(
    fa_out::validation::bytesForFrames(
      "audio.chunk_duration_ms", std::numeric_limits<size_t>::max(), 2),
    std::invalid_argument);
}

TEST(FaOutAudioConfigValidation, RejectsAlsaPluginPcmSinkIds)
{
  EXPECT_NO_THROW(fa_out::validation::requireRawAlsaHardwareSink("hw:1,0"));
  EXPECT_NO_THROW(fa_out::validation::requireRawAlsaHardwareSink("hw:CARD=Device,DEV=0"));

  EXPECT_THROW(fa_out::validation::requireRawAlsaHardwareSink(""), std::invalid_argument);
  EXPECT_THROW(fa_out::validation::requireRawAlsaHardwareSink("default"), std::invalid_argument);
  EXPECT_THROW(fa_out::validation::requireRawAlsaHardwareSink("plughw:1,0"), std::invalid_argument);
  EXPECT_THROW(fa_out::validation::requireRawAlsaHardwareSink("pulse"), std::invalid_argument);
  EXPECT_THROW(fa_out::validation::requireRawAlsaHardwareSink("pipewire"), std::invalid_argument);
}

}  // namespace

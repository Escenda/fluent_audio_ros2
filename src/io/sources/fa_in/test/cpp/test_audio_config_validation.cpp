#include "fa_in/audio_config_validation.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace
{

TEST(FaInAudioConfigValidation, RejectsNonPositiveAndOutOfRangeUint32)
{
  EXPECT_EQ(fa_in::validation::requirePositiveUint32("audio.sample_rate", 48000), 48000u);
  EXPECT_THROW(
    fa_in::validation::requirePositiveUint32("audio.sample_rate", 0),
    std::runtime_error);
  EXPECT_THROW(
    fa_in::validation::requirePositiveUint32("audio.sample_rate", -1),
    std::runtime_error);
  EXPECT_THROW(
    fa_in::validation::requirePositiveUint32(
      "audio.sample_rate",
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1),
    std::runtime_error);
}

TEST(FaInAudioConfigValidation, RejectsNonIntegerCaptureFrameCounts)
{
  EXPECT_EQ(fa_in::validation::captureFramesPerBuffer(48000, 20), 960u);
  EXPECT_THROW(
    fa_in::validation::captureFramesPerBuffer(44100, 1),
    std::runtime_error);
}

TEST(FaInAudioConfigValidation, RejectsByteCountOverflow)
{
  EXPECT_EQ(fa_in::validation::bytesPerFrame(2, 16), 4u);
  EXPECT_THROW(fa_in::validation::bytesPerFrame(1, 7), std::runtime_error);
  EXPECT_THROW(
    fa_in::validation::bytesForFrames(
      "audio.chunk_ms", std::numeric_limits<size_t>::max(), 2),
    std::runtime_error);
}

}  // namespace

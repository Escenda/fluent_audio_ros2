#include <cstring>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_kws/audio_utils.hpp"

namespace
{

fa_interfaces::msg::AudioFrame makeCanonicalFrame()
{
  fa_interfaces::msg::AudioFrame msg;
  msg.source_id = "mic";
  msg.stream_id = "audio/raw/mic";
  msg.sample_rate = 16000;
  msg.channels = 1;
  msg.bit_depth = 32;
  msg.encoding = "FLOAT32LE";
  msg.layout = "interleaved";

  const std::vector<float> samples{0.0F, 0.25F, -0.5F};
  msg.data.resize(samples.size() * sizeof(float));
  std::memcpy(msg.data.data(), samples.data(), msg.data.size());
  return msg;
}

}  // namespace

TEST(AudioUtilsContract, AcceptsCanonicalFloat32AudioFrame)
{
  const auto samples = fa_kws::frameToCanonicalFloat(
    makeCanonicalFrame(), "mic", "audio/raw/mic");

  ASSERT_EQ(samples.size(), 3U);
  EXPECT_FLOAT_EQ(samples[0], 0.0F);
  EXPECT_FLOAT_EQ(samples[1], 0.25F);
  EXPECT_FLOAT_EQ(samples[2], -0.5F);
}

TEST(AudioUtilsContract, RejectsPcm32PayloadBeforeFloatInterpretation)
{
  auto msg = makeCanonicalFrame();
  msg.encoding = "PCM32LE";

  EXPECT_THROW(
    {
      (void)fa_kws::frameToCanonicalFloat(msg, "mic", "audio/raw/mic");
    },
    std::invalid_argument);
}

TEST(AudioUtilsContract, RejectsEmptyPayloadAfterMetadataValidation)
{
  auto msg = makeCanonicalFrame();
  msg.data.clear();

  EXPECT_THROW(
    {
      (void)fa_kws::frameToCanonicalFloat(msg, "mic", "audio/raw/mic");
    },
    std::invalid_argument);
}

TEST(AudioUtilsContract, RejectsBadMetadataBeforeEmptyPayloadPolicy)
{
  auto msg = makeCanonicalFrame();
  msg.data.clear();
  msg.encoding = "PCM32LE";

  EXPECT_THROW(
    {
      (void)fa_kws::frameToCanonicalFloat(msg, "mic", "audio/raw/mic");
    },
    std::invalid_argument);
}

TEST(AudioUtilsContract, RejectsNonNormalizedFloatSamples)
{
  auto msg = makeCanonicalFrame();
  const std::vector<float> samples{1.25F};
  msg.data.resize(samples.size() * sizeof(float));
  std::memcpy(msg.data.data(), samples.data(), msg.data.size());

  EXPECT_THROW(
    {
      (void)fa_kws::frameToCanonicalFloat(msg, "mic", "audio/raw/mic");
    },
    std::invalid_argument);
}

TEST(AudioUtilsContract, RejectsMissingExpectedSourceBinding)
{
  auto msg = makeCanonicalFrame();

  EXPECT_THROW(
    {
      (void)fa_kws::frameToCanonicalFloat(msg, "", "audio/raw/mic");
    },
    std::invalid_argument);
}

TEST(AudioUtilsContract, RejectsMissingExpectedStreamBinding)
{
  auto msg = makeCanonicalFrame();

  EXPECT_THROW(
    {
      (void)fa_kws::frameToCanonicalFloat(msg, "mic", "");
    },
    std::invalid_argument);
}

TEST(AudioUtilsContract, RejectsUnexpectedSourceIdentity)
{
  auto msg = makeCanonicalFrame();

  EXPECT_THROW(
    {
      (void)fa_kws::frameToCanonicalFloat(msg, "system-audio", "audio/raw/mic");
    },
    std::invalid_argument);
}

TEST(AudioUtilsContract, RejectsMissingFrameSourceIdentity)
{
  auto msg = makeCanonicalFrame();
  msg.source_id.clear();

  EXPECT_THROW(
    {
      (void)fa_kws::frameToCanonicalFloat(msg, "mic", "audio/raw/mic");
    },
    std::invalid_argument);
}

TEST(AudioUtilsContract, RejectsMissingFrameStreamIdentity)
{
  auto msg = makeCanonicalFrame();
  msg.stream_id.clear();

  EXPECT_THROW(
    {
      (void)fa_kws::frameToCanonicalFloat(msg, "mic", "audio/raw/mic");
    },
    std::invalid_argument);
}

TEST(AudioUtilsContract, RejectsUnexpectedStreamIdentity)
{
  auto msg = makeCanonicalFrame();

  EXPECT_THROW(
    {
      (void)fa_kws::frameToCanonicalFloat(msg, "mic", "audio/other");
    },
    std::invalid_argument);
}

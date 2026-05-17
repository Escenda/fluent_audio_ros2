#include "fa_resample/resample_core.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace
{

std::vector<uint8_t> float32LeBytes(const std::vector<float> & samples)
{
  std::vector<uint8_t> bytes;
  bytes.reserve(samples.size() * sizeof(float));
  for (const float sample : samples) {
    fa_resample::appendFloat32Le(sample, bytes);
  }
  return bytes;
}

}  // namespace

TEST(ResampleCoreContract, AcceptsOnlyFloat32LeInterleavedFrames)
{
  const fa_resample::FrameContract valid{
    "FLOAT32LE",
    48000,
    1,
    32,
    "interleaved",
    sizeof(float) * 160};

  EXPECT_EQ(
    fa_resample::validateFloat32InterleavedContract(valid),
    fa_resample::FrameContractStatus::kOk);

  auto pcm16 = valid;
  pcm16.encoding = "PCM16LE";
  pcm16.bit_depth = 16;
  EXPECT_EQ(
    fa_resample::validateFloat32InterleavedContract(pcm16),
    fa_resample::FrameContractStatus::kUnsupportedEncoding);

  auto pcm32 = valid;
  pcm32.encoding = "PCM32LE";
  EXPECT_EQ(
    fa_resample::validateFloat32InterleavedContract(pcm32),
    fa_resample::FrameContractStatus::kUnsupportedEncoding);

  auto planar = valid;
  planar.layout = "planar";
  EXPECT_EQ(
    fa_resample::validateFloat32InterleavedContract(planar),
    fa_resample::FrameContractStatus::kUnsupportedLayout);

  auto unaligned = valid;
  unaligned.data_size = sizeof(float) * 2 + 1;
  EXPECT_EQ(
    fa_resample::validateFloat32InterleavedContract(unaligned),
    fa_resample::FrameContractStatus::kUnalignedData);
}

TEST(ResampleCoreContract, DecodesAndEncodesFloat32LeWithoutPcmConversion)
{
  const std::vector<float> samples{-1.0F, -0.5F, 0.0F, 0.5F, 1.0F};
  const std::vector<uint8_t> bytes = float32LeBytes(samples);

  const std::vector<float> decoded = fa_resample::decodeFloat32Le(bytes);
  ASSERT_EQ(decoded.size(), samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    EXPECT_FLOAT_EQ(decoded.at(i), samples.at(i));
  }

  const std::vector<uint8_t> encoded = fa_resample::encodeFloat32Le(decoded);
  EXPECT_EQ(encoded, bytes);
}

TEST(ResampleCoreContract, RejectsNonFiniteOrOutOfRangeSamplesInsteadOfClamping)
{
  EXPECT_FALSE(fa_resample::containsOnlyFiniteNormalizedSamples({0.0F, 1.25F}));
  EXPECT_FALSE(fa_resample::containsOnlyFiniteNormalizedSamples({0.0F, -1.25F}));
  EXPECT_FALSE(fa_resample::containsOnlyFiniteNormalizedSamples({0.0F, NAN}));
  EXPECT_TRUE(fa_resample::containsOnlyFiniteNormalizedSamples({-1.0F, 0.0F, 1.0F}));

  EXPECT_TRUE(fa_resample::encodeFloat32Le({0.0F, 1.25F}).empty());

  uint32_t out_frames = 0;
  EXPECT_TRUE(
    fa_resample::resampleLinear({0.0F, 1.25F}, 48000, 16000, 1, 2, out_frames).empty());
}

TEST(ResampleCoreContract, ResamplesLinearFloat32WithoutChangingChannels)
{
  std::vector<float> input;
  input.reserve(480);
  for (int i = 0; i < 480; ++i) {
    input.push_back(static_cast<float>(i) / 480.0F);
  }

  uint32_t out_frames = 0;
  const std::vector<float> output =
    fa_resample::resampleLinear(input, 48000, 16000, 1, 480, out_frames);

  EXPECT_EQ(out_frames, 160U);
  ASSERT_EQ(output.size(), 160U);
  EXPECT_FLOAT_EQ(output.front(), input.front());
  EXPECT_GT(output.back(), 0.0F);
  EXPECT_LT(output.back(), 1.0F);
}

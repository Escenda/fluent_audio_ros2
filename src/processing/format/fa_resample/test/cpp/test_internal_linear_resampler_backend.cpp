#include "fa_resample/backends/internal_linear_resampler.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace
{

std::vector<uint8_t> float32LeBytes(const std::vector<float> & samples)
{
  std::vector<uint8_t> bytes;
  bytes.reserve(samples.size() * sizeof(float));
  for (const float sample : samples) {
    fa_resample::backends::appendFloat32Le(sample, bytes);
  }
  return bytes;
}

fa_resample::backends::FrameContract frameContract(
  uint32_t sample_rate,
  uint32_t channels,
  const std::vector<uint8_t> & bytes)
{
  return fa_resample::backends::FrameContract{
    "FLOAT32LE",
    sample_rate,
    channels,
    32,
    "interleaved",
    bytes.size()};
}

}  // namespace

TEST(InternalLinearResamplerBackendContract, RejectsInvalidConfig)
{
  EXPECT_THROW(
    fa_resample::backends::InternalLinearResamplerBackend(
      fa_resample::backends::InternalLinearResamplerConfig{0}),
    std::runtime_error);
}

TEST(InternalLinearResamplerBackendContract, AcceptsOnlyFloat32LeInterleavedFrames)
{
  const fa_resample::backends::FrameContract valid{
    "FLOAT32LE",
    48000,
    1,
    32,
    "interleaved",
    sizeof(float) * 160};

  EXPECT_EQ(
    fa_resample::backends::validateFloat32InterleavedContract(valid),
    fa_resample::backends::FrameContractStatus::kOk);

  auto pcm16 = valid;
  pcm16.encoding = "PCM16LE";
  pcm16.bit_depth = 16;
  EXPECT_EQ(
    fa_resample::backends::validateFloat32InterleavedContract(pcm16),
    fa_resample::backends::FrameContractStatus::kUnsupportedEncoding);

  auto pcm32 = valid;
  pcm32.encoding = "PCM32LE";
  EXPECT_EQ(
    fa_resample::backends::validateFloat32InterleavedContract(pcm32),
    fa_resample::backends::FrameContractStatus::kUnsupportedEncoding);

  auto planar = valid;
  planar.layout = "planar";
  EXPECT_EQ(
    fa_resample::backends::validateFloat32InterleavedContract(planar),
    fa_resample::backends::FrameContractStatus::kUnsupportedLayout);

  auto unaligned = valid;
  unaligned.data_size = sizeof(float) * 2 + 1;
  EXPECT_EQ(
    fa_resample::backends::validateFloat32InterleavedContract(unaligned),
    fa_resample::backends::FrameContractStatus::kUnalignedData);
}

TEST(InternalLinearResamplerBackendContract, DecodesAndEncodesFloat32LeWithoutPcmConversion)
{
  const std::vector<float> samples{-1.0F, -0.5F, 0.0F, 0.5F, 1.0F};
  const std::vector<uint8_t> bytes = float32LeBytes(samples);

  const std::vector<float> decoded = fa_resample::backends::decodeFloat32Le(bytes);
  ASSERT_EQ(decoded.size(), samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    EXPECT_FLOAT_EQ(decoded.at(i), samples.at(i));
  }

  const std::vector<uint8_t> encoded = fa_resample::backends::encodeFloat32Le(decoded);
  EXPECT_EQ(encoded, bytes);
}

TEST(InternalLinearResamplerBackendContract, RejectsNonFiniteOrOutOfRangeSamplesInsteadOfClamping)
{
  EXPECT_FALSE(fa_resample::backends::containsOnlyFiniteNormalizedSamples({0.0F, 1.25F}));
  EXPECT_FALSE(fa_resample::backends::containsOnlyFiniteNormalizedSamples({0.0F, -1.25F}));
  EXPECT_FALSE(fa_resample::backends::containsOnlyFiniteNormalizedSamples({0.0F, NAN}));
  EXPECT_TRUE(fa_resample::backends::containsOnlyFiniteNormalizedSamples({-1.0F, 0.0F, 1.0F}));

  EXPECT_TRUE(fa_resample::backends::encodeFloat32Le({0.0F, 1.25F}).empty());

  uint32_t out_frames = 0;
  EXPECT_TRUE(
    fa_resample::backends::resampleLinear({0.0F, 1.25F}, 48000, 16000, 1, 2, out_frames)
    .empty());
}

TEST(InternalLinearResamplerBackendContract, ResamplesLinearFloat32WithoutChangingChannels)
{
  std::vector<float> input;
  input.reserve(480);
  for (int i = 0; i < 480; ++i) {
    input.push_back(static_cast<float>(i) / 480.0F);
  }

  uint32_t out_frames = 0;
  const std::vector<float> output =
    fa_resample::backends::resampleLinear(input, 48000, 16000, 1, 480, out_frames);

  EXPECT_EQ(out_frames, 160U);
  ASSERT_EQ(output.size(), 160U);
  EXPECT_FLOAT_EQ(output.front(), input.front());
  EXPECT_GT(output.back(), 0.0F);
  EXPECT_LT(output.back(), 1.0F);
}

TEST(InternalLinearResamplerBackendContract, ResamplesToConfiguredHigherSampleRate)
{
  const std::vector<float> input{0.0F, 0.25F, 0.5F, 0.75F, 1.0F};

  uint32_t out_frames = 0;
  const std::vector<float> output =
    fa_resample::backends::resampleLinear(input, 16000, 48000, 1, 5, out_frames);

  EXPECT_EQ(out_frames, 15U);
  ASSERT_EQ(output.size(), 15U);
  EXPECT_FLOAT_EQ(output.front(), input.front());
  EXPECT_LE(output.back(), 1.0F);
}

TEST(InternalLinearResamplerBackendContract, ReturnsTypedStatusAndLeavesOutputOnFailure)
{
  fa_resample::backends::InternalLinearResamplerBackend backend(
    fa_resample::backends::InternalLinearResamplerConfig{16000});

  std::vector<uint8_t> output = float32LeBytes({0.125F});
  const std::vector<uint8_t> invalid_samples = float32LeBytes({1.25F});
  const fa_resample::backends::ProcessResult result = backend.process(
    invalid_samples,
    frameContract(48000, 1, invalid_samples),
    output);

  EXPECT_EQ(result.status, fa_resample::backends::ProcessStatus::kInvalidInputSamples);
  EXPECT_EQ(output, float32LeBytes({0.125F}));
}

TEST(InternalLinearResamplerBackendContract, ProcessesBytesThroughBackend)
{
  fa_resample::backends::InternalLinearResamplerBackend backend(
    fa_resample::backends::InternalLinearResamplerConfig{16000});

  std::vector<float> samples;
  samples.reserve(480);
  for (int i = 0; i < 480; ++i) {
    samples.push_back(static_cast<float>(i) / 480.0F);
  }
  const std::vector<uint8_t> input = float32LeBytes(samples);

  std::vector<uint8_t> output;
  const fa_resample::backends::ProcessResult result =
    backend.process(input, frameContract(48000, 1, input), output);

  EXPECT_EQ(result.status, fa_resample::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.output_frames, 160U);
  EXPECT_EQ(output.size(), 160U * sizeof(float));
}

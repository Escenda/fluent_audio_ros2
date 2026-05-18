#include "fa_channel_convert/backends/internal_float32le_channel_convert.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace
{

fa_channel_convert::backends::InternalFloat32LeChannelConvertConfig monoToStereoConfig()
{
  return fa_channel_convert::backends::InternalFloat32LeChannelConvertConfig{
    1,
    2,
    "mono_to_stereo_duplicate",
    16000,
    "FLOAT32LE",
    32,
    "interleaved"};
}

fa_channel_convert::backends::InternalFloat32LeChannelConvertConfig stereoToMonoConfig()
{
  return fa_channel_convert::backends::InternalFloat32LeChannelConvertConfig{
    2,
    1,
    "stereo_to_mono_average",
    16000,
    "FLOAT32LE",
    32,
    "interleaved"};
}

fa_channel_convert::backends::FrameContract validContract(
  const uint32_t channels,
  const size_t data_size)
{
  return fa_channel_convert::backends::FrameContract{
    "FLOAT32LE",
    32,
    16000,
    channels,
    "interleaved",
    data_size};
}

std::vector<uint8_t> float32Bytes(const std::vector<float> & samples)
{
  std::vector<uint8_t> bytes;
  for (const float sample : samples) {
    fa_channel_convert::backends::appendFloat32Le(sample, bytes);
  }
  return bytes;
}

}  // namespace

TEST(InternalFloat32LeChannelConvertBackend, SupportsOnlyExplicitChannelConversions)
{
  EXPECT_TRUE(
    fa_channel_convert::backends::isSupportedChannelConversion(
      "mono_to_stereo_duplicate", 1, 2));
  EXPECT_TRUE(
    fa_channel_convert::backends::isSupportedChannelConversion(
      "stereo_to_mono_average", 2, 1));
  EXPECT_FALSE(
    fa_channel_convert::backends::isSupportedChannelConversion(
      "mono_to_stereo_duplicate", 1, 1));
  EXPECT_FALSE(
    fa_channel_convert::backends::isSupportedChannelConversion(
      "stereo_to_mono_average", 1, 2));

  auto unsupported = monoToStereoConfig();
  unsupported.output_channels = 1;
  EXPECT_THROW({
    fa_channel_convert::backends::InternalFloat32LeChannelConvertBackend backend(unsupported);
    (void)backend;
  }, std::runtime_error);
}

TEST(InternalFloat32LeChannelConvertBackend, DuplicatesMonoToStereo)
{
  fa_channel_convert::backends::InternalFloat32LeChannelConvertBackend backend(monoToStereoConfig());
  const std::vector<uint8_t> input = float32Bytes({-1.0F, 0.25F, 1.0F});
  std::vector<uint8_t> output;

  const fa_channel_convert::backends::ProcessResult result =
    backend.process(input, validContract(1, input.size()), output);

  EXPECT_EQ(result.status, fa_channel_convert::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.frames, 3U);
  ASSERT_EQ(output.size(), 6U * sizeof(float));
  EXPECT_FLOAT_EQ(fa_channel_convert::backends::readFloat32Le(output, 0), -1.0F);
  EXPECT_FLOAT_EQ(fa_channel_convert::backends::readFloat32Le(output, 1), -1.0F);
  EXPECT_FLOAT_EQ(fa_channel_convert::backends::readFloat32Le(output, 2), 0.25F);
  EXPECT_FLOAT_EQ(fa_channel_convert::backends::readFloat32Le(output, 3), 0.25F);
  EXPECT_FLOAT_EQ(fa_channel_convert::backends::readFloat32Le(output, 4), 1.0F);
  EXPECT_FLOAT_EQ(fa_channel_convert::backends::readFloat32Le(output, 5), 1.0F);
}

TEST(InternalFloat32LeChannelConvertBackend, AveragesStereoToMono)
{
  fa_channel_convert::backends::InternalFloat32LeChannelConvertBackend backend(stereoToMonoConfig());
  const std::vector<uint8_t> input = float32Bytes({-1.0F, 1.0F, 0.25F, 0.75F});
  std::vector<uint8_t> output;

  const fa_channel_convert::backends::ProcessResult result =
    backend.process(input, validContract(2, input.size()), output);

  EXPECT_EQ(result.status, fa_channel_convert::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.frames, 2U);
  ASSERT_EQ(output.size(), 2U * sizeof(float));
  EXPECT_FLOAT_EQ(fa_channel_convert::backends::readFloat32Le(output, 0), 0.0F);
  EXPECT_FLOAT_EQ(fa_channel_convert::backends::readFloat32Le(output, 1), 0.5F);
}

TEST(InternalFloat32LeChannelConvertBackend, RejectsInvalidFloat32SamplesWithoutMutatingOutput)
{
  fa_channel_convert::backends::InternalFloat32LeChannelConvertBackend backend(monoToStereoConfig());
  const std::vector<uint8_t> out_of_range_input = float32Bytes({1.0001F});
  std::vector<uint8_t> output = {0xaaU};

  const fa_channel_convert::backends::ProcessResult out_of_range =
    backend.process(out_of_range_input, validContract(1, out_of_range_input.size()), output);

  EXPECT_EQ(
    out_of_range.status,
    fa_channel_convert::backends::ProcessStatus::kOutOfRangeFloat32Input);
  EXPECT_EQ(output, (std::vector<uint8_t>{0xaaU}));

  const std::vector<uint8_t> non_finite_input =
    float32Bytes({std::numeric_limits<float>::quiet_NaN()});
  const fa_channel_convert::backends::ProcessResult non_finite =
    backend.process(non_finite_input, validContract(1, non_finite_input.size()), output);

  EXPECT_EQ(
    non_finite.status,
    fa_channel_convert::backends::ProcessStatus::kNonFiniteFloat32Input);
  EXPECT_EQ(output, (std::vector<uint8_t>{0xaaU}));
}

TEST(InternalFloat32LeChannelConvertBackend, RejectsMalformedContractWithoutMutatingOutput)
{
  fa_channel_convert::backends::InternalFloat32LeChannelConvertBackend backend(monoToStereoConfig());
  const std::vector<uint8_t> input = {0x00U};
  std::vector<uint8_t> output = {0xaaU};

  const fa_channel_convert::backends::ProcessResult result =
    backend.process(input, validContract(1, input.size()), output);

  EXPECT_EQ(result.status, fa_channel_convert::backends::ProcessStatus::kInvalidFrameContract);
  EXPECT_EQ(
    result.frame_contract_status,
    fa_channel_convert::backends::FrameContractStatus::kUnalignedData);
  EXPECT_EQ(output, (std::vector<uint8_t>{0xaaU}));
}

#include "fa_denoise/backends/denoise_backend.hpp"
#include "fa_denoise/backends/passthrough_backend.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace
{
fa_denoise::backends::AudioFormat pcm16Mono()
{
  fa_denoise::backends::AudioFormat format;
  format.sample_rate = 16000;
  format.channels = 1;
  format.encoding = fa_denoise::backends::kEncodingPcm16;
  format.bit_depth = 16;
  return format;
}

fa_denoise::backends::AudioFormat float32Mono()
{
  fa_denoise::backends::AudioFormat format;
  format.sample_rate = 16000;
  format.channels = 1;
  format.encoding = fa_denoise::backends::kEncodingFloat32;
  format.bit_depth = 32;
  return format;
}
}  // namespace

TEST(DenoiseBackendContractTest, DecodePcm16ToNormalizedFloat)
{
  fa_denoise::backends::AudioBuffer input;
  input.format = pcm16Mono();
  input.data = {0x00U, 0x80U, 0x00U, 0x00U, 0xffU, 0x7fU};

  std::vector<float> output;
  const auto status = fa_denoise::backends::decodeToFloat(input, output);

  ASSERT_EQ(status, fa_denoise::backends::ProcessStatus::kOk);
  ASSERT_EQ(output.size(), 3U);
  EXPECT_FLOAT_EQ(output[0], -1.0f);
  EXPECT_FLOAT_EQ(output[1], 0.0f);
  EXPECT_NEAR(output[2], 32767.0f / 32768.0f, 1.0e-7f);
}

TEST(DenoiseBackendContractTest, DecodeFloat32RejectsOutOfRangeWithoutOverwritingOutput)
{
  const float out_of_range = 1.25f;
  fa_denoise::backends::AudioBuffer input;
  input.format = float32Mono();
  input.data.resize(sizeof(float));
  std::memcpy(input.data.data(), &out_of_range, sizeof(float));

  std::vector<float> output{0.5f};
  const auto status = fa_denoise::backends::decodeToFloat(input, output);

  EXPECT_EQ(status, fa_denoise::backends::ProcessStatus::kOutOfRangeInput);
  ASSERT_EQ(output.size(), 1U);
  EXPECT_FLOAT_EQ(output[0], 0.5f);
}

TEST(DenoiseBackendContractTest, EncodeFloatToPcm16RejectsOverflowWithoutOverwritingOutput)
{
  std::vector<uint8_t> output{0x42U};
  const auto status = fa_denoise::backends::encodeFromFloat(
    {1.01f},
    pcm16Mono(),
    output);

  EXPECT_EQ(status, fa_denoise::backends::ProcessStatus::kOutOfRangeOutput);
  EXPECT_EQ(output, std::vector<uint8_t>{0x42U});
}

TEST(DenoiseBackendContractTest, RejectsUnhandledStatusValues)
{
  EXPECT_THROW(
    (void)fa_denoise::backends::processStatusMessage(
      static_cast<fa_denoise::backends::ProcessStatus>(999)),
    std::logic_error);
}

TEST(DenoiseBackendContractTest, PassthroughRequiresMatchingFormat)
{
  auto output_format = pcm16Mono();
  output_format.encoding = fa_denoise::backends::kEncodingFloat32;
  output_format.bit_depth = 32;

  EXPECT_THROW(
    (fa_denoise::backends::PassthroughBackend(pcm16Mono(), output_format)),
    std::runtime_error);
}

TEST(DenoiseBackendContractTest, PassthroughCopiesBytesWithoutRosMessageKnowledge)
{
  fa_denoise::backends::PassthroughBackend backend(pcm16Mono(), pcm16Mono());
  fa_denoise::backends::AudioBuffer input;
  input.format = pcm16Mono();
  input.data = {0x01U, 0x02U, 0x03U, 0x04U};

  const fa_denoise::backends::ProcessResult result = backend.process(input);

  ASSERT_EQ(result.status, fa_denoise::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.output.format.encoding, fa_denoise::backends::kEncodingPcm16);
  EXPECT_EQ(result.output.format.bit_depth, 16);
  EXPECT_EQ(result.output.data, input.data);
}

#include "fa_sample_format/sample_format_conversion.hpp"

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace
{

float readFloat32Le(const std::vector<uint8_t> & bytes, const size_t index)
{
  const size_t offset = index * sizeof(float);
  float value = 0.0F;
  std::memcpy(&value, bytes.data() + offset, sizeof(float));
  return value;
}

}  // namespace

TEST(SampleFormatConversion, SupportsOnlyExplicitIntegerPcmToFloat32Le)
{
  EXPECT_TRUE(fa_sample_format::isSupportedSampleFormatConversion("PCM16LE", 16, "FLOAT32LE", 32));
  EXPECT_TRUE(fa_sample_format::isSupportedSampleFormatConversion("PCM32LE", 32, "FLOAT32LE", 32));
  EXPECT_FALSE(fa_sample_format::isSupportedSampleFormatConversion("PCM16LE", 16, "PCM16LE", 16));
  EXPECT_FALSE(fa_sample_format::isSupportedSampleFormatConversion("FLOAT32LE", 32, "FLOAT32LE", 32));
}

TEST(SampleFormatConversion, ConvertsPcm16LeToFloat32LeWithoutClipping)
{
  const std::vector<uint8_t> input = {
    0x00, 0x80,  // -32768
    0x00, 0x00,  // 0
    0xFF, 0x7F,  // 32767
  };

  const std::vector<uint8_t> output = fa_sample_format::convertPcm16ToFloat32(input);

  ASSERT_EQ(output.size(), 3U * sizeof(float));
  EXPECT_FLOAT_EQ(readFloat32Le(output, 0), -1.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(output, 1), 0.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(output, 2), 32767.0F / 32768.0F);
}

TEST(SampleFormatConversion, ConvertsPcm32LeToFloat32LeWithoutClipping)
{
  const std::vector<uint8_t> input = {
    0x00, 0x00, 0x00, 0x80,  // -2147483648
    0x00, 0x00, 0x00, 0x00,  // 0
    0xFF, 0xFF, 0xFF, 0x7F,  // 2147483647
  };

  const std::vector<uint8_t> output = fa_sample_format::convertPcm32ToFloat32(input);

  ASSERT_EQ(output.size(), 3U * sizeof(float));
  EXPECT_FLOAT_EQ(readFloat32Le(output, 0), -1.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(output, 1), 0.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(output, 2), 2147483647.0F / 2147483648.0F);
}

TEST(SampleFormatConversion, RejectsMalformedPayloads)
{
  EXPECT_TRUE(fa_sample_format::convertPcm16ToFloat32({}).empty());
  EXPECT_TRUE(fa_sample_format::convertPcm16ToFloat32({0x00}).empty());
  EXPECT_TRUE(fa_sample_format::convertPcm32ToFloat32({0x00, 0x00, 0x00}).empty());
}

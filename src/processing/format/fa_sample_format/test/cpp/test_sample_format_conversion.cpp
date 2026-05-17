#include "fa_sample_format/sample_format_conversion.hpp"

#include <cstdint>
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

int16_t readPcm16Le(const std::vector<uint8_t> & bytes, const size_t index)
{
  const size_t offset = index * sizeof(int16_t);
  const uint16_t raw =
    static_cast<uint16_t>(bytes.at(offset)) |
    (static_cast<uint16_t>(bytes.at(offset + 1)) << 8U);
  return static_cast<int16_t>(raw);
}

}  // namespace

TEST(SampleFormatConversion, SupportsOnlyExplicitIntegerPcmToFloat32Le)
{
  EXPECT_TRUE(fa_sample_format::isSupportedSampleFormatConversion("PCM16LE", 16, "FLOAT32LE", 32));
  EXPECT_TRUE(fa_sample_format::isSupportedSampleFormatConversion("PCM32LE", 32, "FLOAT32LE", 32));
  EXPECT_TRUE(fa_sample_format::isSupportedSampleFormatConversion("FLOAT32LE", 32, "PCM16LE", 16));
  EXPECT_FALSE(fa_sample_format::isSupportedSampleFormatConversion("PCM16LE", 16, "PCM16LE", 16));
  EXPECT_FALSE(fa_sample_format::isSupportedSampleFormatConversion("FLOAT32LE", 32, "FLOAT32LE", 32));
  EXPECT_FALSE(fa_sample_format::isSupportedSampleFormatConversion("FLOAT32LE", 32, "PCM32LE", 32));
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

TEST(SampleFormatConversion, ConvertsFloat32LeToPcm16LeWithoutClipping)
{
  std::vector<uint8_t> input;
  fa_sample_format::appendFloat32Le(-1.0F, input);
  fa_sample_format::appendFloat32Le(0.0F, input);
  fa_sample_format::appendFloat32Le(1.0F, input);

  const std::vector<uint8_t> output = fa_sample_format::convertFloat32ToPcm16(input);

  ASSERT_EQ(output.size(), 3U * sizeof(int16_t));
  EXPECT_EQ(readPcm16Le(output, 0), -32768);
  EXPECT_EQ(readPcm16Le(output, 1), 0);
  EXPECT_EQ(readPcm16Le(output, 2), 32767);
}

TEST(SampleFormatConversion, RejectsFloat32LeToPcm16LeOutOfRangeWithoutClipping)
{
  std::vector<uint8_t> input;
  fa_sample_format::appendFloat32Le(1.0001F, input);

  EXPECT_TRUE(fa_sample_format::convertFloat32ToPcm16(input).empty());
}

TEST(SampleFormatConversion, RejectsMalformedPayloads)
{
  EXPECT_TRUE(fa_sample_format::convertPcm16ToFloat32({}).empty());
  EXPECT_TRUE(fa_sample_format::convertPcm16ToFloat32({0x00}).empty());
  EXPECT_TRUE(fa_sample_format::convertPcm32ToFloat32({0x00, 0x00, 0x00}).empty());
  EXPECT_TRUE(fa_sample_format::convertFloat32ToPcm16({0x00, 0x00, 0x00}).empty());
}

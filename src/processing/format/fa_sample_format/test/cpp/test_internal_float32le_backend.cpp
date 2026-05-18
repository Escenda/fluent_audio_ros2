#include "fa_sample_format/backends/internal_float32le.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace
{

fa_sample_format::backends::InternalFloat32LeConfig validPcm16ToFloat32Config()
{
  return fa_sample_format::backends::InternalFloat32LeConfig{
    "PCM16LE",
    16,
    "FLOAT32LE",
    32,
    16000,
    1,
    "interleaved"};
}

fa_sample_format::backends::FrameContract validContract(
  const std::string & encoding,
  const uint32_t bit_depth,
  const size_t data_size)
{
  return fa_sample_format::backends::FrameContract{
    encoding,
    bit_depth,
    16000,
    1,
    "interleaved",
    data_size};
}

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

TEST(InternalFloat32LeBackend, SupportsOnlyExplicitConfiguredConversions)
{
  EXPECT_TRUE(
    fa_sample_format::backends::isSupportedSampleFormatConversion(
      "PCM16LE", 16, "FLOAT32LE", 32));
  EXPECT_TRUE(
    fa_sample_format::backends::isSupportedSampleFormatConversion(
      "PCM32LE", 32, "FLOAT32LE", 32));
  EXPECT_TRUE(
    fa_sample_format::backends::isSupportedSampleFormatConversion(
      "FLOAT32LE", 32, "PCM16LE", 16));
  EXPECT_FALSE(
    fa_sample_format::backends::isSupportedSampleFormatConversion(
      "FLOAT32LE", 32, "FLOAT32LE", 32));

  auto unsupported = validPcm16ToFloat32Config();
  unsupported.output_encoding = "PCM32LE";
  unsupported.output_bit_depth = 32;
  EXPECT_THROW({
    fa_sample_format::backends::InternalFloat32LeBackend backend(unsupported);
    (void)backend;
  }, std::runtime_error);
}

TEST(InternalFloat32LeBackend, ConvertsPcm16LeToFloat32LeWithoutClipping)
{
  fa_sample_format::backends::InternalFloat32LeBackend backend(validPcm16ToFloat32Config());
  const std::vector<uint8_t> input = {
    0x00, 0x80,
    0x00, 0x00,
    0xff, 0x7f,
  };
  std::vector<uint8_t> output;

  const fa_sample_format::backends::ProcessResult result =
    backend.process(input, validContract("PCM16LE", 16, input.size()), output);

  EXPECT_EQ(result.status, fa_sample_format::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples, 3U);
  ASSERT_EQ(output.size(), 3U * sizeof(float));
  EXPECT_FLOAT_EQ(readFloat32Le(output, 0), -1.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(output, 1), 0.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(output, 2), 32767.0F / 32768.0F);
}

TEST(InternalFloat32LeBackend, ConvertsPcm32LeToFloat32LeWithoutClipping)
{
  auto config = validPcm16ToFloat32Config();
  config.input_encoding = "PCM32LE";
  config.input_bit_depth = 32;
  fa_sample_format::backends::InternalFloat32LeBackend backend(config);
  const std::vector<uint8_t> input = {
    0x00, 0x00, 0x00, 0x80,
    0x00, 0x00, 0x00, 0x00,
    0xff, 0xff, 0xff, 0x7f,
  };
  std::vector<uint8_t> output;

  const fa_sample_format::backends::ProcessResult result =
    backend.process(input, validContract("PCM32LE", 32, input.size()), output);

  EXPECT_EQ(result.status, fa_sample_format::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples, 3U);
  ASSERT_EQ(output.size(), 3U * sizeof(float));
  EXPECT_FLOAT_EQ(readFloat32Le(output, 0), -1.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(output, 1), 0.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(output, 2), 2147483647.0F / 2147483648.0F);
}

TEST(InternalFloat32LeBackend, ConvertsFloat32LeToPcm16LeWithoutClipping)
{
  auto config = validPcm16ToFloat32Config();
  config.input_encoding = "FLOAT32LE";
  config.input_bit_depth = 32;
  config.output_encoding = "PCM16LE";
  config.output_bit_depth = 16;
  fa_sample_format::backends::InternalFloat32LeBackend backend(config);
  std::vector<uint8_t> input;
  fa_sample_format::backends::appendFloat32Le(-1.0F, input);
  fa_sample_format::backends::appendFloat32Le(0.0F, input);
  fa_sample_format::backends::appendFloat32Le(1.0F, input);
  std::vector<uint8_t> output;

  const fa_sample_format::backends::ProcessResult result =
    backend.process(input, validContract("FLOAT32LE", 32, input.size()), output);

  EXPECT_EQ(result.status, fa_sample_format::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples, 3U);
  ASSERT_EQ(output.size(), 3U * sizeof(int16_t));
  EXPECT_EQ(readPcm16Le(output, 0), -32768);
  EXPECT_EQ(readPcm16Le(output, 1), 0);
  EXPECT_EQ(readPcm16Le(output, 2), 32767);
}

TEST(InternalFloat32LeBackend, RejectsInvalidFloat32SamplesWithoutMutatingOutput)
{
  auto config = validPcm16ToFloat32Config();
  config.input_encoding = "FLOAT32LE";
  config.input_bit_depth = 32;
  config.output_encoding = "PCM16LE";
  config.output_bit_depth = 16;
  fa_sample_format::backends::InternalFloat32LeBackend backend(config);
  std::vector<uint8_t> input;
  fa_sample_format::backends::appendFloat32Le(1.0001F, input);
  std::vector<uint8_t> output = {0xaaU};

  const fa_sample_format::backends::ProcessResult out_of_range =
    backend.process(input, validContract("FLOAT32LE", 32, input.size()), output);

  EXPECT_EQ(
    out_of_range.status,
    fa_sample_format::backends::ProcessStatus::kOutOfRangeFloat32Input);
  EXPECT_EQ(output, (std::vector<uint8_t>{0xaaU}));

  input.clear();
  fa_sample_format::backends::appendFloat32Le(
    std::numeric_limits<float>::quiet_NaN(),
    input);
  const fa_sample_format::backends::ProcessResult non_finite =
    backend.process(input, validContract("FLOAT32LE", 32, input.size()), output);

  EXPECT_EQ(
    non_finite.status,
    fa_sample_format::backends::ProcessStatus::kNonFiniteFloat32Input);
  EXPECT_EQ(output, (std::vector<uint8_t>{0xaaU}));
}

TEST(InternalFloat32LeBackend, RejectsMalformedContractWithoutMutatingOutput)
{
  fa_sample_format::backends::InternalFloat32LeBackend backend(validPcm16ToFloat32Config());
  std::vector<uint8_t> output = {0xaaU};

  const fa_sample_format::backends::ProcessResult result =
    backend.process({0x00U}, validContract("PCM16LE", 16, 1), output);

  EXPECT_EQ(result.status, fa_sample_format::backends::ProcessStatus::kInvalidFrameContract);
  EXPECT_EQ(
    result.frame_contract_status,
    fa_sample_format::backends::FrameContractStatus::kUnalignedData);
  EXPECT_EQ(output, (std::vector<uint8_t>{0xaaU}));
}

#include "fa_bit_depth/backends/internal_integer_bit_depth.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace
{

fa_bit_depth::backends::InternalIntegerBitDepthConfig validConfig()
{
  return fa_bit_depth::backends::InternalIntegerBitDepthConfig{
    "PCM16LE",
    16,
    "PCM32LE",
    32,
    16000,
    1,
    "interleaved"};
}

fa_bit_depth::backends::FrameContract validContract(const size_t data_size)
{
  return fa_bit_depth::backends::FrameContract{
    "PCM16LE",
    16,
    16000,
    1,
    "interleaved",
    data_size};
}

}  // namespace

TEST(InternalIntegerBitDepthBackend, AcceptsOnlyLosslessPcm16LeToPcm32LeConfig)
{
  EXPECT_TRUE(fa_bit_depth::backends::isSupportedConversion("PCM16LE", 16, "PCM32LE", 32));
  EXPECT_FALSE(fa_bit_depth::backends::isSupportedConversion("PCM32LE", 32, "PCM16LE", 16));
  EXPECT_FALSE(fa_bit_depth::backends::isSupportedConversion("FLOAT32LE", 32, "PCM16LE", 16));

  auto unsupported = validConfig();
  unsupported.input_encoding = "PCM32LE";
  unsupported.input_bit_depth = 32;
  unsupported.output_encoding = "PCM16LE";
  unsupported.output_bit_depth = 16;
  EXPECT_THROW({
    fa_bit_depth::backends::InternalIntegerBitDepthBackend backend(unsupported);
    (void)backend;
  }, std::runtime_error);
}

TEST(InternalIntegerBitDepthBackend, RejectsUnhandledStatusValues)
{
  EXPECT_THROW(
    fa_bit_depth::backends::frameContractStatusName(
      static_cast<fa_bit_depth::backends::FrameContractStatus>(999)),
    std::logic_error);
  EXPECT_THROW(
    fa_bit_depth::backends::processStatusMessage(
      static_cast<fa_bit_depth::backends::ProcessStatus>(999)),
    std::logic_error);
}

TEST(InternalIntegerBitDepthBackend, ConvertsPcm16LeWordsToPcm32LeHighWords)
{
  fa_bit_depth::backends::InternalIntegerBitDepthBackend backend(validConfig());
  const std::vector<uint8_t> input = {
    0x00U, 0x00U,
    0xffU, 0x7fU,
    0x00U, 0x80U,
    0xffU, 0xffU,
  };
  std::vector<uint8_t> output;

  const fa_bit_depth::backends::ProcessResult result =
    backend.process(input, validContract(input.size()), output);

  EXPECT_EQ(result.status, fa_bit_depth::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.frame_contract_status, fa_bit_depth::backends::FrameContractStatus::kOk);
  EXPECT_EQ(result.samples, 4U);
  EXPECT_EQ(
    output,
    std::vector<uint8_t>({
      0x00U, 0x00U, 0x00U, 0x00U,
      0x00U, 0x00U, 0xffU, 0x7fU,
      0x00U, 0x00U, 0x00U, 0x80U,
      0x00U, 0x00U, 0xffU, 0xffU,
    }));
}

TEST(InternalIntegerBitDepthBackend, RejectsMisalignedContractWithoutMutatingOutput)
{
  fa_bit_depth::backends::InternalIntegerBitDepthBackend backend(validConfig());
  std::vector<uint8_t> output = {0xaaU, 0xbbU};

  const fa_bit_depth::backends::ProcessResult result =
    backend.process({0x00U}, validContract(1), output);

  EXPECT_EQ(result.status, fa_bit_depth::backends::ProcessStatus::kInvalidFrameContract);
  EXPECT_EQ(result.frame_contract_status, fa_bit_depth::backends::FrameContractStatus::kUnalignedData);
  EXPECT_EQ(output, (std::vector<uint8_t>{0xaaU, 0xbbU}));
}

TEST(InternalIntegerBitDepthBackend, RejectsRuntimeMetadataMismatch)
{
  fa_bit_depth::backends::InternalIntegerBitDepthBackend backend(validConfig());
  auto contract = validContract(2);
  contract.input_encoding = "PCM32LE";

  EXPECT_EQ(
    backend.validateContract(contract),
    fa_bit_depth::backends::FrameContractStatus::kUnsupportedInputEncoding);
}

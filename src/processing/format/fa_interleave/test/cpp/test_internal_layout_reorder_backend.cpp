#include "fa_interleave/backends/internal_layout_reorder.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace
{

fa_interleave::backends::InternalLayoutReorderConfig interleavedToPlanarConfig()
{
  return fa_interleave::backends::InternalLayoutReorderConfig{
    "interleaved",
    "planar",
    16000,
    2,
    "PCM16LE",
    16};
}

fa_interleave::backends::FrameContract contract(
  const std::string & layout,
  const size_t data_size)
{
  return fa_interleave::backends::FrameContract{
    layout,
    "PCM16LE",
    16,
    16000,
    2,
    data_size};
}

}  // namespace

TEST(InternalLayoutReorderBackend, AcceptsOnlyExplicitLayoutAndFormatContracts)
{
  EXPECT_TRUE(fa_interleave::backends::isSupportedLayoutConversion("interleaved", "planar"));
  EXPECT_TRUE(fa_interleave::backends::isSupportedLayoutConversion("planar", "interleaved"));
  EXPECT_FALSE(fa_interleave::backends::isSupportedLayoutConversion("interleaved", "interleaved"));
  EXPECT_TRUE(fa_interleave::backends::isSupportedFormat("FLOAT32LE", 32));
  EXPECT_TRUE(fa_interleave::backends::isSupportedFormat("PCM16LE", 16));
  EXPECT_TRUE(fa_interleave::backends::isSupportedFormat("PCM32LE", 32));
  EXPECT_FALSE(fa_interleave::backends::isSupportedFormat("PCM24LE", 24));

  auto invalid = interleavedToPlanarConfig();
  invalid.output_layout = "interleaved";
  EXPECT_THROW({
    fa_interleave::backends::InternalLayoutReorderBackend backend(invalid);
    (void)backend;
  }, std::runtime_error);
}

TEST(InternalLayoutReorderBackend, ReordersInterleavedToPlanarBySampleBytes)
{
  fa_interleave::backends::InternalLayoutReorderBackend backend(interleavedToPlanarConfig());
  const std::vector<uint8_t> input = {
    0x10U, 0x00U, 0x20U, 0x00U,
    0x11U, 0x00U, 0x21U, 0x00U,
    0x12U, 0x00U, 0x22U, 0x00U,
  };
  std::vector<uint8_t> output;

  const fa_interleave::backends::ProcessResult result =
    backend.process(input, contract("interleaved", input.size()), output);

  EXPECT_EQ(result.status, fa_interleave::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.frames, 3U);
  EXPECT_EQ(
    output,
    std::vector<uint8_t>({
      0x10U, 0x00U, 0x11U, 0x00U, 0x12U, 0x00U,
      0x20U, 0x00U, 0x21U, 0x00U, 0x22U, 0x00U,
    }));
}

TEST(InternalLayoutReorderBackend, ReordersPlanarToInterleavedBySampleBytes)
{
  auto config = interleavedToPlanarConfig();
  config.input_layout = "planar";
  config.output_layout = "interleaved";
  fa_interleave::backends::InternalLayoutReorderBackend backend(config);
  const std::vector<uint8_t> input = {
    0x10U, 0x00U, 0x11U, 0x00U, 0x12U, 0x00U,
    0x20U, 0x00U, 0x21U, 0x00U, 0x22U, 0x00U,
  };
  std::vector<uint8_t> output;

  const fa_interleave::backends::ProcessResult result =
    backend.process(input, contract("planar", input.size()), output);

  EXPECT_EQ(result.status, fa_interleave::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.frames, 3U);
  EXPECT_EQ(
    output,
    std::vector<uint8_t>({
      0x10U, 0x00U, 0x20U, 0x00U,
      0x11U, 0x00U, 0x21U, 0x00U,
      0x12U, 0x00U, 0x22U, 0x00U,
    }));
}

TEST(InternalLayoutReorderBackend, RejectsMisalignedPayloadWithoutMutatingOutput)
{
  fa_interleave::backends::InternalLayoutReorderBackend backend(interleavedToPlanarConfig());
  std::vector<uint8_t> output = {0xaaU};

  const fa_interleave::backends::ProcessResult result =
    backend.process({0x10U, 0x00U}, contract("interleaved", 2), output);

  EXPECT_EQ(result.status, fa_interleave::backends::ProcessStatus::kInvalidFrameContract);
  EXPECT_EQ(
    result.frame_contract_status,
    fa_interleave::backends::FrameContractStatus::kUnalignedData);
  EXPECT_EQ(output, (std::vector<uint8_t>{0xaaU}));
}

TEST(InternalLayoutReorderBackend, RejectsRuntimeMetadataMismatch)
{
  fa_interleave::backends::InternalLayoutReorderBackend backend(interleavedToPlanarConfig());
  auto mismatched = contract("planar", 4);

  EXPECT_EQ(
    backend.validateContract(mismatched),
    fa_interleave::backends::FrameContractStatus::kUnsupportedInputLayout);
}

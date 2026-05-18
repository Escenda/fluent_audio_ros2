#include "fa_decode/backends/external_codec_decoder.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace
{

fa_decode::backends::ExternalCodecDecoderConfig baseConfig()
{
  return fa_decode::backends::ExternalCodecDecoderConfig{
    "/bin/cat",
    {},
    1000,
    1024,
    "opus",
    "ogg",
    "ogg_page",
    16000,
    1,
    16000,
    1,
    "PCM16LE",
    16,
    "interleaved"};
}

fa_decode::backends::EncodedChunkContract baseContract(const size_t data_size)
{
  return fa_decode::backends::EncodedChunkContract{
    "opus",
    "ogg",
    "ogg_page",
    16000,
    1,
    20000000,
    data_size};
}

}  // namespace

TEST(ExternalCodecDecoderBackendContract, ValidatesStartupConfig)
{
  auto config = baseConfig();
  config.executable.clear();
  EXPECT_THROW({
    fa_decode::backends::ExternalCodecDecoderBackend backend(config);
  }, std::runtime_error);

  config = baseConfig();
  config.input_codec.clear();
  EXPECT_THROW({
    fa_decode::backends::ExternalCodecDecoderBackend backend(config);
  }, std::runtime_error);

  config = baseConfig();
  config.output_sample_rate = 48000;
  EXPECT_THROW({
    fa_decode::backends::ExternalCodecDecoderBackend backend(config);
  }, std::runtime_error);
}

TEST(ExternalCodecDecoderBackendContract, RejectsUnhandledStatusValues)
{
  EXPECT_THROW(
    fa_decode::backends::encodedChunkContractStatusName(
      static_cast<fa_decode::backends::EncodedChunkContractStatus>(999)),
    std::logic_error);
  EXPECT_THROW(
    fa_decode::backends::decodeStatusMessage(
      static_cast<fa_decode::backends::DecodeStatus>(999)),
    std::logic_error);
}

TEST(ExternalCodecDecoderBackendContract, DecodesViaExplicitExternalCommand)
{
  fa_decode::backends::ExternalCodecDecoderBackend backend(baseConfig());
  const std::vector<uint8_t> input{0x01, 0x00, 0x02, 0x00};

  const fa_decode::backends::DecodeResult result =
    backend.decode(input, baseContract(input.size()));

  EXPECT_EQ(result.status, fa_decode::backends::DecodeStatus::kOk);
  EXPECT_EQ(
    result.chunk_contract_status,
    fa_decode::backends::EncodedChunkContractStatus::kOk);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.encoding, "PCM16LE");
  EXPECT_EQ(result.bit_depth, 16U);
  EXPECT_EQ(result.sample_rate, 16000U);
  EXPECT_EQ(result.channels, 1U);
  EXPECT_EQ(result.layout, "interleaved");
  EXPECT_EQ(result.data, input);
}

TEST(ExternalCodecDecoderBackendContract, RejectsMismatchedChunkContractBeforeCommand)
{
  fa_decode::backends::ExternalCodecDecoderBackend backend(baseConfig());
  auto contract = baseContract(4);
  contract.codec = "aac";

  const fa_decode::backends::DecodeResult result =
    backend.decode(std::vector<uint8_t>{0x00, 0x00, 0x00, 0x00}, contract);

  EXPECT_EQ(result.status, fa_decode::backends::DecodeStatus::kInvalidChunkContract);
  EXPECT_EQ(
    result.chunk_contract_status,
    fa_decode::backends::EncodedChunkContractStatus::kInvalidCodec);
}

TEST(ExternalCodecDecoderBackendContract, ReportsExternalCommandFailures)
{
  auto config = baseConfig();
  config.executable = "/bin/false";
  fa_decode::backends::ExternalCodecDecoderBackend backend(config);
  const std::vector<uint8_t> input{0x01, 0x00};

  const fa_decode::backends::DecodeResult result =
    backend.decode(input, baseContract(input.size()));

  EXPECT_EQ(result.status, fa_decode::backends::DecodeStatus::kCommandFailed);
  EXPECT_NE(result.exit_code, 0);
}

TEST(ExternalCodecDecoderBackendContract, RejectsEmptyOversizedAndUnalignedOutput)
{
  auto config = baseConfig();
  config.executable = "/bin/true";
  fa_decode::backends::ExternalCodecDecoderBackend empty_backend(config);
  const std::vector<uint8_t> input{0x01, 0x00};

  EXPECT_EQ(
    empty_backend.decode(input, baseContract(input.size())).status,
    fa_decode::backends::DecodeStatus::kEmptyOutput);

  config = baseConfig();
  config.max_output_bytes = 1;
  fa_decode::backends::ExternalCodecDecoderBackend oversized_backend(config);
  EXPECT_EQ(
    oversized_backend.decode(input, baseContract(input.size())).status,
    fa_decode::backends::DecodeStatus::kOutputTooLarge);

  config = baseConfig();
  fa_decode::backends::ExternalCodecDecoderBackend unaligned_backend(config);
  const std::vector<uint8_t> unaligned_input{0x01, 0x00, 0x02};
  EXPECT_EQ(
    unaligned_backend.decode(unaligned_input, baseContract(unaligned_input.size())).status,
    fa_decode::backends::DecodeStatus::kUnalignedOutput);
}

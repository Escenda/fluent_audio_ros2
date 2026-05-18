#include "fa_encode/backends/external_codec_encoder.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace
{

fa_encode::backends::ExternalCodecEncoderConfig baseConfig()
{
  return fa_encode::backends::ExternalCodecEncoderConfig{
    "/bin/cat",
    {},
    1000,
    1024,
    16000,
    1,
    "PCM16LE",
    16,
    "interleaved",
    "opus",
    "ogg",
    "ogg_page"};
}

fa_encode::backends::PcmFrameContract baseContract(const size_t data_size)
{
  return fa_encode::backends::PcmFrameContract{
    "PCM16LE",
    16,
    16000,
    1,
    "interleaved",
    data_size};
}

}  // namespace

TEST(ExternalCodecEncoderBackendContract, ValidatesStartupConfig)
{
  auto config = baseConfig();
  config.executable.clear();
  EXPECT_THROW({
    fa_encode::backends::ExternalCodecEncoderBackend backend(config);
  }, std::runtime_error);

  config = baseConfig();
  config.output_codec.clear();
  EXPECT_THROW({
    fa_encode::backends::ExternalCodecEncoderBackend backend(config);
  }, std::runtime_error);
}

TEST(ExternalCodecEncoderBackendContract, RejectsUnhandledStatusValues)
{
  EXPECT_THROW(
    fa_encode::backends::frameContractStatusName(
      static_cast<fa_encode::backends::FrameContractStatus>(999)),
    std::logic_error);
  EXPECT_THROW(
    fa_encode::backends::encodeStatusMessage(
      static_cast<fa_encode::backends::EncodeStatus>(999)),
    std::logic_error);
}

TEST(ExternalCodecEncoderBackendContract, EncodesViaExplicitExternalCommand)
{
  fa_encode::backends::ExternalCodecEncoderBackend backend(baseConfig());
  const std::vector<uint8_t> input{0x01, 0x00, 0x02, 0x00};

  const fa_encode::backends::EncodeResult result =
    backend.encode(input, baseContract(input.size()));

  EXPECT_EQ(result.status, fa_encode::backends::EncodeStatus::kOk);
  EXPECT_EQ(result.frame_contract_status, fa_encode::backends::FrameContractStatus::kOk);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.codec, "opus");
  EXPECT_EQ(result.container, "ogg");
  EXPECT_EQ(result.payload_format, "ogg_page");
  EXPECT_EQ(result.sample_rate, 16000U);
  EXPECT_EQ(result.channels, 1U);
  EXPECT_EQ(result.data, input);
}

TEST(ExternalCodecEncoderBackendContract, RejectsMismatchedFrameContractBeforeCommand)
{
  fa_encode::backends::ExternalCodecEncoderBackend backend(baseConfig());
  auto contract = baseContract(4);
  contract.encoding = "FLOAT32LE";

  const fa_encode::backends::EncodeResult result =
    backend.encode(std::vector<uint8_t>{0x00, 0x00, 0x00, 0x00}, contract);

  EXPECT_EQ(result.status, fa_encode::backends::EncodeStatus::kInvalidFrameContract);
  EXPECT_EQ(
    result.frame_contract_status,
    fa_encode::backends::FrameContractStatus::kUnsupportedInputEncoding);
}

TEST(ExternalCodecEncoderBackendContract, ReportsExternalCommandFailures)
{
  auto config = baseConfig();
  config.executable = "/bin/false";
  fa_encode::backends::ExternalCodecEncoderBackend backend(config);
  const std::vector<uint8_t> input{0x01, 0x00};

  const fa_encode::backends::EncodeResult result =
    backend.encode(input, baseContract(input.size()));

  EXPECT_EQ(result.status, fa_encode::backends::EncodeStatus::kCommandFailed);
  EXPECT_NE(result.exit_code, 0);
}

TEST(ExternalCodecEncoderBackendContract, RejectsEmptyAndOversizedOutput)
{
  auto config = baseConfig();
  config.executable = "/bin/true";
  fa_encode::backends::ExternalCodecEncoderBackend empty_backend(config);
  const std::vector<uint8_t> input{0x01, 0x00};

  EXPECT_EQ(
    empty_backend.encode(input, baseContract(input.size())).status,
    fa_encode::backends::EncodeStatus::kEmptyOutput);

  config = baseConfig();
  config.max_output_bytes = 1;
  fa_encode::backends::ExternalCodecEncoderBackend oversized_backend(config);
  EXPECT_EQ(
    oversized_backend.encode(input, baseContract(input.size())).status,
    fa_encode::backends::EncodeStatus::kOutputTooLarge);
}

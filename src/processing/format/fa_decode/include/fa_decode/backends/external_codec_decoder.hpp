#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_decode::backends
{

inline constexpr const char * kBackendName = "external_codec_decoder";
inline constexpr const char * kEncodingPcm16Le = "PCM16LE";
inline constexpr const char * kEncodingPcm32Le = "PCM32LE";
inline constexpr const char * kEncodingFloat32Le = "FLOAT32LE";
inline constexpr const char * kInterleavedLayout = "interleaved";

enum class EncodedChunkContractStatus
{
  kOk,
  kInvalidCodec,
  kInvalidContainer,
  kInvalidPayloadFormat,
  kInvalidSampleRate,
  kInvalidChannels,
  kInvalidDuration,
  kEmptyData,
};

enum class DecodeStatus
{
  kOk,
  kInvalidChunkContract,
  kEmptyInput,
  kCommandStartFailed,
  kCommandWriteFailed,
  kCommandReadFailed,
  kCommandTimeout,
  kCommandFailed,
  kEmptyOutput,
  kOutputTooLarge,
  kUnalignedOutput,
};

struct EncodedChunkContract
{
  std::string codec;
  std::string container;
  std::string payload_format;
  uint32_t sample_rate{0};
  uint32_t channels{0};
  uint64_t duration_ns{0};
  size_t data_size{0};
};

struct ExternalCodecDecoderConfig
{
  std::string executable;
  std::vector<std::string> arguments;
  int timeout_ms{-1};
  int max_output_bytes{-1};
  std::string input_codec;
  std::string input_container;
  std::string input_payload_format;
  int input_sample_rate{-1};
  int input_channels{-1};
  int output_sample_rate{-1};
  int output_channels{-1};
  std::string output_encoding;
  int output_bit_depth{-1};
  std::string output_layout;
};

struct DecodeResult
{
  DecodeStatus status{DecodeStatus::kOk};
  EncodedChunkContractStatus chunk_contract_status{EncodedChunkContractStatus::kOk};
  int exit_code{-1};
  std::string encoding;
  uint32_t bit_depth{0};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  std::string layout;
  std::vector<uint8_t> data;
};

class ExternalCodecDecoderBackend
{
public:
  explicit ExternalCodecDecoderBackend(const ExternalCodecDecoderConfig & config);

  [[nodiscard]] EncodedChunkContractStatus validateContract(
    const EncodedChunkContract & contract) const;
  [[nodiscard]] DecodeResult decode(
    const std::vector<uint8_t> & input,
    const EncodedChunkContract & contract) const;

private:
  ExternalCodecDecoderConfig config_;
};

const char * encodedChunkContractStatusName(EncodedChunkContractStatus status);
const char * decodeStatusMessage(DecodeStatus status);
bool isSupportedPcmFormat(const std::string & encoding, int bit_depth);
size_t bytesPerSample(int bit_depth);

}  // namespace fa_decode::backends

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_encode::backends
{

inline constexpr const char * kBackendName = "external_codec_encoder";
inline constexpr const char * kEncodingPcm16Le = "PCM16LE";
inline constexpr const char * kEncodingPcm32Le = "PCM32LE";
inline constexpr const char * kEncodingFloat32Le = "FLOAT32LE";
inline constexpr const char * kInterleavedLayout = "interleaved";

enum class FrameContractStatus
{
  kOk,
  kInvalidSampleRate,
  kInvalidChannels,
  kUnsupportedInputEncoding,
  kUnsupportedInputBitDepth,
  kUnsupportedLayout,
  kEmptyData,
  kUnalignedData,
};

enum class EncodeStatus
{
  kOk,
  kInvalidFrameContract,
  kEmptyInput,
  kCommandStartFailed,
  kCommandWriteFailed,
  kCommandReadFailed,
  kCommandTimeout,
  kCommandFailed,
  kEmptyOutput,
  kOutputTooLarge,
};

struct PcmFrameContract
{
  std::string encoding;
  uint32_t bit_depth{0};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  std::string layout;
  size_t data_size{0};
};

struct ExternalCodecEncoderConfig
{
  std::string executable;
  std::vector<std::string> arguments;
  int timeout_ms{-1};
  int max_output_bytes{-1};
  int input_sample_rate{-1};
  int input_channels{-1};
  std::string input_encoding;
  int input_bit_depth{-1};
  std::string input_layout;
  std::string output_codec;
  std::string output_container;
  std::string output_payload_format;
};

struct EncodeResult
{
  EncodeStatus status{EncodeStatus::kOk};
  FrameContractStatus frame_contract_status{FrameContractStatus::kOk};
  int exit_code{-1};
  std::string codec;
  std::string container;
  std::string payload_format;
  uint32_t sample_rate{0};
  uint32_t channels{0};
  std::vector<uint8_t> data;
};

class ExternalCodecEncoderBackend
{
public:
  explicit ExternalCodecEncoderBackend(const ExternalCodecEncoderConfig & config);

  [[nodiscard]] FrameContractStatus validateContract(const PcmFrameContract & contract) const;
  [[nodiscard]] EncodeResult encode(
    const std::vector<uint8_t> & input,
    const PcmFrameContract & contract) const;

private:
  ExternalCodecEncoderConfig config_;
};

const char * frameContractStatusName(FrameContractStatus status);
const char * encodeStatusMessage(EncodeStatus status);
bool isSupportedPcmFormat(const std::string & encoding, int bit_depth);
size_t bytesPerSample(int bit_depth);

}  // namespace fa_encode::backends

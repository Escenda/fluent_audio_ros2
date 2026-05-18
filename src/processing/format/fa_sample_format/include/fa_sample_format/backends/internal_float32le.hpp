#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_sample_format::backends
{

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
  kUnsupportedOutputEncoding,
  kUnsupportedOutputBitDepth,
  kUnsupportedLayout,
  kEmptyData,
  kUnalignedData,
};

enum class ProcessStatus
{
  kOk,
  kInvalidFrameContract,
  kNonFiniteFloat32Input,
  kOutOfRangeFloat32Input,
  kConversionFailed,
};

struct FrameContract
{
  std::string input_encoding;
  uint32_t input_bit_depth{0};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  std::string layout;
  size_t data_size{0};
};

struct InternalFloat32LeConfig
{
  std::string input_encoding;
  int input_bit_depth{-1};
  std::string output_encoding;
  int output_bit_depth{-1};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_layout;
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  FrameContractStatus frame_contract_status{FrameContractStatus::kOk};
  size_t samples{0};
};

struct ByteConversionResult
{
  ProcessStatus status{ProcessStatus::kOk};
  size_t samples{0};
  std::vector<uint8_t> data;
};

class InternalFloat32LeBackend
{
public:
  static constexpr const char * kName = "internal_float32le";

  explicit InternalFloat32LeBackend(const InternalFloat32LeConfig & config);

  [[nodiscard]] const std::string & outputEncoding() const;
  [[nodiscard]] int outputBitDepth() const;
  [[nodiscard]] FrameContractStatus validateContract(const FrameContract & contract) const;
  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    const FrameContract & contract,
    std::vector<uint8_t> & output) const;

private:
  InternalFloat32LeConfig config_;
};

const char * frameContractStatusName(FrameContractStatus status);
const char * processStatusMessage(ProcessStatus status);

bool isSupportedSampleFormatConversion(
  const std::string & input_encoding,
  int input_bit_depth,
  const std::string & output_encoding,
  int output_bit_depth);
size_t bytesPerSample(int bit_depth);
ByteConversionResult convertPcm16ToFloat32(const std::vector<uint8_t> & input_bytes);
ByteConversionResult convertPcm32ToFloat32(const std::vector<uint8_t> & input_bytes);
ByteConversionResult convertFloat32ToPcm16(const std::vector<uint8_t> & input_bytes);
void appendFloat32Le(float sample, std::vector<uint8_t> & out_bytes);
void appendPcm16Le(int16_t sample, std::vector<uint8_t> & out_bytes);

}  // namespace fa_sample_format::backends

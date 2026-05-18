#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_bit_depth::backends
{

inline constexpr const char * kEncodingPcm16Le = "PCM16LE";
inline constexpr const char * kEncodingPcm32Le = "PCM32LE";
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

struct InternalIntegerBitDepthConfig
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

class InternalIntegerBitDepthBackend
{
public:
  static constexpr const char * kName = "internal_integer_bit_depth";

  explicit InternalIntegerBitDepthBackend(const InternalIntegerBitDepthConfig & config);

  [[nodiscard]] const std::string & outputEncoding() const;
  [[nodiscard]] int outputBitDepth() const;
  [[nodiscard]] FrameContractStatus validateContract(const FrameContract & contract) const;
  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    const FrameContract & contract,
    std::vector<uint8_t> & output) const;

private:
  InternalIntegerBitDepthConfig config_;
};

const char * frameContractStatusName(FrameContractStatus status);
const char * processStatusMessage(ProcessStatus status);

bool isSupportedConversion(
  const std::string & input_encoding,
  int input_bit_depth,
  const std::string & output_encoding,
  int output_bit_depth);
size_t bytesPerSample(int bit_depth);
std::vector<uint8_t> convertPcm16ToPcm32(const std::vector<uint8_t> & input_bytes);
void appendPcm32Le(uint32_t sample, std::vector<uint8_t> & out_bytes);

}  // namespace fa_bit_depth::backends

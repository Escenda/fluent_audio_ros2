#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_channel_convert::backends
{

inline constexpr const char * kEncodingFloat32Le = "FLOAT32LE";
inline constexpr const char * kInterleavedLayout = "interleaved";
inline constexpr const char * kModeMonoToStereoDuplicate = "mono_to_stereo_duplicate";
inline constexpr const char * kModeStereoToMonoAverage = "stereo_to_mono_average";

enum class FrameContractStatus
{
  kOk,
  kInvalidSampleRate,
  kInvalidChannels,
  kUnsupportedEncoding,
  kUnsupportedBitDepth,
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
  kUnsupportedConversion,
};

struct FrameContract
{
  std::string encoding;
  uint32_t bit_depth{0};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  std::string layout;
  size_t data_size{0};
};

struct InternalFloat32LeChannelConvertConfig
{
  int input_channels{-1};
  int output_channels{-1};
  std::string conversion_mode;
  int expected_sample_rate{-1};
  std::string expected_encoding;
  int expected_bit_depth{-1};
  std::string expected_layout;
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  FrameContractStatus frame_contract_status{FrameContractStatus::kOk};
  size_t frames{0};
};

struct ChannelConversionResult
{
  ProcessStatus status{ProcessStatus::kOk};
  size_t frames{0};
  std::vector<uint8_t> data;
};

class InternalFloat32LeChannelConvertBackend
{
public:
  static constexpr const char * kName = "internal_float32le_channel_convert";

  explicit InternalFloat32LeChannelConvertBackend(
    const InternalFloat32LeChannelConvertConfig & config);

  [[nodiscard]] int outputChannels() const;
  [[nodiscard]] FrameContractStatus validateContract(const FrameContract & contract) const;
  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    const FrameContract & contract,
    std::vector<uint8_t> & output) const;

private:
  InternalFloat32LeChannelConvertConfig config_;
};

const char * frameContractStatusName(FrameContractStatus status);
const char * processStatusMessage(ProcessStatus status);

bool isSupportedChannelConversion(const std::string & mode, int input_channels, int output_channels);
float readFloat32Le(const std::vector<uint8_t> & bytes, size_t sample_index);
void appendFloat32Le(float sample, std::vector<uint8_t> & out_bytes);
bool isNormalizedFinite(float sample);
ChannelConversionResult convertMonoToStereoDuplicate(
  const std::vector<uint8_t> & input_bytes,
  size_t frame_count);
ChannelConversionResult convertStereoToMonoAverage(
  const std::vector<uint8_t> & input_bytes,
  size_t frame_count);

}  // namespace fa_channel_convert::backends

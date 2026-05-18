#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_interleave::backends
{

inline constexpr const char * kEncodingPcm16Le = "PCM16LE";
inline constexpr const char * kEncodingPcm32Le = "PCM32LE";
inline constexpr const char * kEncodingFloat32Le = "FLOAT32LE";
inline constexpr const char * kInterleavedLayout = "interleaved";
inline constexpr const char * kPlanarLayout = "planar";

enum class FrameContractStatus
{
  kOk,
  kInvalidSampleRate,
  kInvalidChannels,
  kUnsupportedInputLayout,
  kUnsupportedEncoding,
  kUnsupportedBitDepth,
  kEmptyData,
  kUnalignedData,
};

enum class ProcessStatus
{
  kOk,
  kInvalidFrameContract,
  kReorderFailed,
};

struct FrameContract
{
  std::string layout;
  std::string encoding;
  uint32_t bit_depth{0};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  size_t data_size{0};
};

struct InternalLayoutReorderConfig
{
  std::string input_layout;
  std::string output_layout;
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding;
  int expected_bit_depth{-1};
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  FrameContractStatus frame_contract_status{FrameContractStatus::kOk};
  size_t frames{0};
};

class InternalLayoutReorderBackend
{
public:
  static constexpr const char * kName = "internal_layout_reorder";

  explicit InternalLayoutReorderBackend(const InternalLayoutReorderConfig & config);

  [[nodiscard]] const std::string & outputLayout() const;
  [[nodiscard]] FrameContractStatus validateContract(const FrameContract & contract) const;
  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    const FrameContract & contract,
    std::vector<uint8_t> & output) const;

private:
  InternalLayoutReorderConfig config_;
};

const char * frameContractStatusName(FrameContractStatus status);
const char * processStatusMessage(ProcessStatus status);

bool isSupportedLayout(const std::string & layout);
bool isSupportedLayoutConversion(const std::string & input_layout, const std::string & output_layout);
bool isSupportedFormat(const std::string & encoding, int bit_depth);
size_t bytesPerSample(const std::string & encoding, int bit_depth);
std::vector<uint8_t> reorderInterleavedToPlanar(
  const std::vector<uint8_t> & input_bytes,
  size_t frame_count,
  size_t channel_count,
  size_t bytes_per_sample);
std::vector<uint8_t> reorderPlanarToInterleaved(
  const std::vector<uint8_t> & input_bytes,
  size_t frame_count,
  size_t channel_count,
  size_t bytes_per_sample);
void appendSampleBytes(
  const std::vector<uint8_t> & input_bytes,
  size_t sample_index,
  size_t bytes_per_sample,
  std::vector<uint8_t> & output_bytes);

}  // namespace fa_interleave::backends

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_resample::backends
{

inline constexpr const char * kEncodingFloat32Le = "FLOAT32LE";
inline constexpr const char * kInterleavedLayout = "interleaved";

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

struct FrameContract
{
  std::string encoding;
  uint32_t sample_rate{0};
  uint32_t channels{0};
  uint32_t bit_depth{0};
  std::string layout;
  size_t data_size{0};
};

enum class ProcessStatus
{
  kOk,
  kInvalidFrameContract,
  kInvalidInputSamples,
  kResampleFailed,
  kEncodeFailed,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  FrameContractStatus frame_contract_status{FrameContractStatus::kOk};
  uint32_t output_frames{0};
};

struct InternalLinearResamplerConfig
{
  int target_sample_rate{-1};
};

class InternalLinearResamplerBackend
{
public:
  static constexpr const char * kName = "internal_linear_resampler";

  explicit InternalLinearResamplerBackend(const InternalLinearResamplerConfig & config);

  [[nodiscard]] int targetSampleRate() const;
  [[nodiscard]] ProcessResult process(
    const std::vector<uint8_t> & input,
    const FrameContract & contract,
    std::vector<uint8_t> & output) const;

private:
  InternalLinearResamplerConfig config_;
};

const char * frameContractStatusName(FrameContractStatus status);
const char * processStatusMessage(ProcessStatus status);

FrameContractStatus validateFloat32InterleavedContract(const FrameContract & contract);
bool containsOnlyFiniteNormalizedSamples(const std::vector<float> & samples);
std::vector<float> decodeFloat32Le(const std::vector<uint8_t> & bytes);
void appendFloat32Le(float sample, std::vector<uint8_t> & out_bytes);
std::vector<uint8_t> encodeFloat32Le(const std::vector<float> & samples);
std::vector<float> resampleLinear(
  const std::vector<float> & interleaved,
  uint32_t in_rate,
  uint32_t out_rate,
  uint32_t channels,
  uint32_t in_frames,
  uint32_t & out_frames);

}  // namespace fa_resample::backends

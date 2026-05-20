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
  kInvalidStreamIdentity,
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

struct StreamContract
{
  std::string stream_id;
  FrameContract frame;
  int target_sample_rate{-1};
  std::string backend_name;
  std::string backend_quality;
};

enum class ProcessStatus
{
  kOk,
  kInvalidFrameContract,
  kInvalidInputSamples,
  kStreamContractViolation,
  kBackendProcessFailed,
  kEncodeFailed,
};

struct ProcessResult
{
  ProcessStatus status{ProcessStatus::kOk};
  FrameContractStatus frame_contract_status{FrameContractStatus::kOk};
  uint32_t output_frames{0};
};

struct ProcessRequest
{
  const std::string & stream_id;
  const std::vector<uint8_t> & input;
  FrameContract contract;
};

struct BackendMetrics
{
  double algorithmic_delay_input_samples{0.0};
  double algorithmic_delay_output_samples{0.0};
  double algorithmic_delay_ms{0.0};
  uint64_t process_call_count{0};
  uint64_t processing_time_total_ns{0};
  uint64_t processing_time_max_ns{0};
  uint64_t input_frames_total{0};
  uint64_t output_frames_total{0};
  double expected_output_frames{0.0};
  int64_t frame_count_error_samples{0};
};

class ResamplerBackend
{
public:
  virtual ~ResamplerBackend() = default;

  [[nodiscard]] virtual std::string name() const = 0;
  [[nodiscard]] virtual std::string quality() const = 0;
  [[nodiscard]] virtual int targetSampleRate() const = 0;
  [[nodiscard]] virtual BackendMetrics metrics() const = 0;
  [[nodiscard]] virtual ProcessResult process(
    const ProcessRequest & request,
    std::vector<uint8_t> & output) = 0;
};

const char * frameContractStatusName(FrameContractStatus status);
const char * processStatusMessage(ProcessStatus status);

FrameContractStatus validateFloat32InterleavedContract(const FrameContract & contract);
bool streamContractMatches(const StreamContract & current, const StreamContract & next);
StreamContract makeStreamContract(
  const std::string & stream_id,
  const FrameContract & frame,
  int target_sample_rate,
  const std::string & backend_name,
  const std::string & backend_quality);

bool containsOnlyFiniteNormalizedSamples(const std::vector<float> & samples);
std::vector<float> decodeFloat32Le(const std::vector<uint8_t> & bytes);
void appendFloat32Le(float sample, std::vector<uint8_t> & out_bytes);
std::vector<uint8_t> encodeFloat32Le(const std::vector<float> & samples);
uint32_t frameCountFromContract(const FrameContract & contract);
double processingTimeMeanMs(const BackendMetrics & metrics);
double processingTimeMaxMs(const BackendMetrics & metrics);
void recordProcessingTime(BackendMetrics & metrics, uint64_t elapsed_ns);
void updateFrameCountMetrics(
  BackendMetrics & metrics,
  uint64_t input_frames_total,
  uint64_t output_frames_total,
  uint32_t input_rate,
  uint32_t output_rate);

}  // namespace fa_resample::backends

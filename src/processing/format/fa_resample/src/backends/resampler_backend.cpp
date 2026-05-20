#include "fa_resample/backends/resampler_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fa_resample::backends
{

const char * frameContractStatusName(const FrameContractStatus status)
{
  switch (status) {
    case FrameContractStatus::kOk:
      return "ok";
    case FrameContractStatus::kInvalidStreamIdentity:
      return "invalid_stream_identity";
    case FrameContractStatus::kInvalidSampleRate:
      return "invalid_sample_rate";
    case FrameContractStatus::kInvalidChannels:
      return "invalid_channels";
    case FrameContractStatus::kUnsupportedEncoding:
      return "unsupported_encoding";
    case FrameContractStatus::kUnsupportedBitDepth:
      return "unsupported_bit_depth";
    case FrameContractStatus::kUnsupportedLayout:
      return "unsupported_layout";
    case FrameContractStatus::kEmptyData:
      return "empty_data";
    case FrameContractStatus::kUnalignedData:
      return "unaligned_data";
  }
  throw std::logic_error("unhandled resample frame contract status");
}

const char * processStatusMessage(const ProcessStatus status)
{
  switch (status) {
    case ProcessStatus::kOk:
      return "ok";
    case ProcessStatus::kInvalidFrameContract:
      return "invalid FLOAT32LE frame contract";
    case ProcessStatus::kInvalidInputSamples:
      return "input samples must be finite normalized [-1.0, 1.0]";
    case ProcessStatus::kStreamContractViolation:
      return "stream contract changed after backend state was created";
    case ProcessStatus::kBackendProcessFailed:
      return "resampler backend process failed";
    case ProcessStatus::kEncodeFailed:
      return "FLOAT32LE encoding failed";
  }
  throw std::logic_error("unhandled resample backend process status");
}

FrameContractStatus validateFloat32InterleavedContract(const FrameContract & contract)
{
  if (contract.sample_rate == 0) {
    return FrameContractStatus::kInvalidSampleRate;
  }
  if (contract.channels == 0) {
    return FrameContractStatus::kInvalidChannels;
  }
  if (contract.encoding != kEncodingFloat32Le) {
    return FrameContractStatus::kUnsupportedEncoding;
  }
  if (contract.bit_depth != 32) {
    return FrameContractStatus::kUnsupportedBitDepth;
  }
  if (contract.layout != kInterleavedLayout) {
    return FrameContractStatus::kUnsupportedLayout;
  }
  if (contract.data_size == 0) {
    return FrameContractStatus::kEmptyData;
  }

  const size_t bytes_per_frame = static_cast<size_t>(contract.channels) * sizeof(float);
  if (bytes_per_frame == 0 || (contract.data_size % bytes_per_frame) != 0) {
    return FrameContractStatus::kUnalignedData;
  }
  return FrameContractStatus::kOk;
}

bool streamContractMatches(const StreamContract & current, const StreamContract & next)
{
  return current.stream_id == next.stream_id &&
         current.frame.encoding == next.frame.encoding &&
         current.frame.sample_rate == next.frame.sample_rate &&
         current.frame.channels == next.frame.channels &&
         current.frame.bit_depth == next.frame.bit_depth &&
         current.frame.layout == next.frame.layout &&
         current.target_sample_rate == next.target_sample_rate &&
         current.backend_name == next.backend_name &&
         current.backend_quality == next.backend_quality;
}

StreamContract makeStreamContract(
  const std::string & stream_id,
  const FrameContract & frame,
  const int target_sample_rate,
  const std::string & backend_name,
  const std::string & backend_quality)
{
  return StreamContract{stream_id, frame, target_sample_rate, backend_name, backend_quality};
}

namespace
{
bool isFiniteNormalizedSample(const float value)
{
  return std::isfinite(static_cast<double>(value)) && value >= -1.0F && value <= 1.0F;
}
}  // namespace

bool containsOnlyFiniteNormalizedSamples(const std::vector<float> & samples)
{
  if (samples.empty()) {
    return false;
  }
  for (const float value : samples) {
    if (!isFiniteNormalizedSample(value)) {
      return false;
    }
  }
  return true;
}

std::vector<float> decodeFloat32Le(const std::vector<uint8_t> & bytes)
{
  if (bytes.empty() || (bytes.size() % sizeof(float)) != 0) {
    return {};
  }

  std::vector<float> samples(bytes.size() / sizeof(float));
  for (size_t byte_index = 0, sample_index = 0; byte_index < bytes.size();
    byte_index += sizeof(float), ++sample_index)
  {
    const uint32_t raw =
      static_cast<uint32_t>(bytes.at(byte_index)) |
      (static_cast<uint32_t>(bytes.at(byte_index + 1)) << 8U) |
      (static_cast<uint32_t>(bytes.at(byte_index + 2)) << 16U) |
      (static_cast<uint32_t>(bytes.at(byte_index + 3)) << 24U);
    std::memcpy(&samples.at(sample_index), &raw, sizeof(float));
  }
  return samples;
}

void appendFloat32Le(const float sample, std::vector<uint8_t> & out_bytes)
{
  uint32_t raw = 0;
  std::memcpy(&raw, &sample, sizeof(float));
  out_bytes.push_back(static_cast<uint8_t>(raw & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 8U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 16U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 24U) & 0xFFU));
}

std::vector<uint8_t> encodeFloat32Le(const std::vector<float> & samples)
{
  if (!containsOnlyFiniteNormalizedSamples(samples)) {
    return {};
  }

  std::vector<uint8_t> out_bytes;
  out_bytes.reserve(samples.size() * sizeof(float));
  for (const float sample : samples) {
    appendFloat32Le(sample, out_bytes);
  }
  return out_bytes;
}

uint32_t frameCountFromContract(const FrameContract & contract)
{
  const size_t bytes_per_frame = static_cast<size_t>(contract.channels) * sizeof(float);
  if (bytes_per_frame == 0) {
    return 0;
  }
  const size_t frame_count = contract.data_size / bytes_per_frame;
  if (frame_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return 0;
  }
  return static_cast<uint32_t>(frame_count);
}

double processingTimeMeanMs(const BackendMetrics & metrics)
{
  if (metrics.process_call_count == 0) {
    return 0.0;
  }
  const double mean_ns =
    static_cast<double>(metrics.processing_time_total_ns) /
    static_cast<double>(metrics.process_call_count);
  return mean_ns / 1000000.0;
}

double processingTimeMaxMs(const BackendMetrics & metrics)
{
  return static_cast<double>(metrics.processing_time_max_ns) / 1000000.0;
}

void recordProcessingTime(BackendMetrics & metrics, const uint64_t elapsed_ns)
{
  metrics.process_call_count += 1;
  metrics.processing_time_total_ns += elapsed_ns;
  metrics.processing_time_max_ns = std::max(metrics.processing_time_max_ns, elapsed_ns);
}

void updateFrameCountMetrics(
  BackendMetrics & metrics,
  const uint64_t input_frames_total,
  const uint64_t output_frames_total,
  const uint32_t input_rate,
  const uint32_t output_rate)
{
  metrics.input_frames_total = input_frames_total;
  metrics.output_frames_total = output_frames_total;
  if (input_rate == 0 || output_rate == 0) {
    metrics.expected_output_frames = 0.0;
    metrics.frame_count_error_samples = 0;
    return;
  }

  metrics.expected_output_frames =
    static_cast<double>(input_frames_total) *
    static_cast<double>(output_rate) /
    static_cast<double>(input_rate);
  metrics.frame_count_error_samples = static_cast<int64_t>(
    std::llround(static_cast<double>(output_frames_total) - metrics.expected_output_frames));
}

}  // namespace fa_resample::backends

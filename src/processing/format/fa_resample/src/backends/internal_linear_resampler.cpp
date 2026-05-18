#include "fa_resample/backends/internal_linear_resampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_resample::backends
{

const char * frameContractStatusName(const FrameContractStatus status)
{
  switch (status) {
    case FrameContractStatus::kOk:
      return "ok";
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
  return "unknown";
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
    case ProcessStatus::kResampleFailed:
      return "linear resampling failed";
    case ProcessStatus::kEncodeFailed:
      return "FLOAT32LE encoding failed";
  }
  return "unknown resample backend status";
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

std::vector<float> resampleLinear(
  const std::vector<float> & interleaved,
  const uint32_t in_rate,
  const uint32_t out_rate,
  const uint32_t channels,
  const uint32_t in_frames,
  uint32_t & out_frames)
{
  out_frames = 0;
  if (in_rate == 0 || out_rate == 0 || channels == 0 || in_frames == 0 || interleaved.empty()) {
    return {};
  }
  if (interleaved.size() != static_cast<size_t>(in_frames) * channels) {
    return {};
  }
  if (!containsOnlyFiniteNormalizedSamples(interleaved)) {
    return {};
  }

  if (in_rate == out_rate) {
    out_frames = in_frames;
    return interleaved;
  }

  const double ratio = static_cast<double>(out_rate) / static_cast<double>(in_rate);
  const double out_frames_f = static_cast<double>(in_frames) * ratio;
  const uint32_t frames = static_cast<uint32_t>(std::max<double>(1.0, std::lround(out_frames_f)));

  std::vector<float> out;
  out.resize(static_cast<size_t>(frames) * channels);

  const double step = static_cast<double>(in_rate) / static_cast<double>(out_rate);
  for (uint32_t i = 0; i < frames; ++i) {
    const double src_pos = static_cast<double>(i) * step;
    uint32_t idx0 = static_cast<uint32_t>(std::floor(src_pos));
    const double frac = src_pos - static_cast<double>(idx0);

    if (idx0 >= in_frames) {
      idx0 = in_frames - 1;
    }
    const uint32_t idx1 = std::min<uint32_t>(idx0 + 1, in_frames - 1);

    for (uint32_t ch = 0; ch < channels; ++ch) {
      const float s0 = interleaved.at(static_cast<size_t>(idx0) * channels + ch);
      const float s1 = interleaved.at(static_cast<size_t>(idx1) * channels + ch);
      const float value = static_cast<float>(
        (1.0 - frac) * static_cast<double>(s0) + frac * static_cast<double>(s1));
      out.at(static_cast<size_t>(i) * channels + ch) = value;
    }
  }

  if (!containsOnlyFiniteNormalizedSamples(out)) {
    return {};
  }

  out_frames = frames;
  return out;
}

InternalLinearResamplerBackend::InternalLinearResamplerBackend(
  const InternalLinearResamplerConfig & config)
: config_(config)
{
  if (config_.target_sample_rate <= 0) {
    throw std::runtime_error("target_sample_rate must be > 0");
  }
}

int InternalLinearResamplerBackend::targetSampleRate() const
{
  return config_.target_sample_rate;
}

ProcessResult InternalLinearResamplerBackend::process(
  const std::vector<uint8_t> & input,
  const FrameContract & contract,
  std::vector<uint8_t> & output) const
{
  const FrameContractStatus frame_contract_status = validateFloat32InterleavedContract(contract);
  if (frame_contract_status != FrameContractStatus::kOk) {
    return ProcessResult{ProcessStatus::kInvalidFrameContract, frame_contract_status, 0};
  }

  const std::vector<float> in_f32 = decodeFloat32Le(input);
  if (!containsOnlyFiniteNormalizedSamples(in_f32)) {
    return ProcessResult{ProcessStatus::kInvalidInputSamples, FrameContractStatus::kOk, 0};
  }

  const uint32_t in_frames = static_cast<uint32_t>(
    input.size() / (static_cast<size_t>(contract.channels) * sizeof(float)));
  uint32_t out_frames = 0;
  const std::vector<float> out_f32 = resampleLinear(
    in_f32,
    contract.sample_rate,
    static_cast<uint32_t>(config_.target_sample_rate),
    contract.channels,
    in_frames,
    out_frames);

  if (out_f32.empty() || out_frames == 0) {
    return ProcessResult{ProcessStatus::kResampleFailed, FrameContractStatus::kOk, 0};
  }

  const std::vector<uint8_t> out_bytes = encodeFloat32Le(out_f32);
  if (out_bytes.empty()) {
    return ProcessResult{ProcessStatus::kEncodeFailed, FrameContractStatus::kOk, 0};
  }

  output = std::move(out_bytes);
  return ProcessResult{ProcessStatus::kOk, FrameContractStatus::kOk, out_frames};
}

}  // namespace fa_resample::backends

#include "fa_channel_convert/backends/internal_float32le_channel_convert.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_channel_convert::backends
{

namespace
{
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;
}  // namespace

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
  throw std::logic_error("unhandled channel-convert frame contract status");
}

const char * processStatusMessage(const ProcessStatus status)
{
  switch (status) {
    case ProcessStatus::kOk:
      return "ok";
    case ProcessStatus::kInvalidFrameContract:
      return "invalid channel-convert frame contract";
    case ProcessStatus::kNonFiniteFloat32Input:
      return "FLOAT32LE input sample is not finite";
    case ProcessStatus::kOutOfRangeFloat32Input:
      return "FLOAT32LE input sample is outside normalized range";
    case ProcessStatus::kUnsupportedConversion:
      return "unsupported channel conversion";
  }
  throw std::logic_error("unhandled channel-convert backend process status");
}

bool isSupportedChannelConversion(
  const std::string & mode,
  const int input_channels,
  const int output_channels)
{
  return (mode == kModeMonoToStereoDuplicate && input_channels == 1 && output_channels == 2) ||
         (mode == kModeStereoToMonoAverage && input_channels == 2 && output_channels == 1);
}

float readFloat32Le(const std::vector<uint8_t> & bytes, const size_t sample_index)
{
  const size_t offset = sample_index * sizeof(float);
  const uint32_t raw =
    static_cast<uint32_t>(bytes.at(offset)) |
    (static_cast<uint32_t>(bytes.at(offset + 1U)) << 8U) |
    (static_cast<uint32_t>(bytes.at(offset + 2U)) << 16U) |
    (static_cast<uint32_t>(bytes.at(offset + 3U)) << 24U);
  float sample = 0.0F;
  std::memcpy(&sample, &raw, sizeof(float));
  return sample;
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

bool isNormalizedFinite(const float sample)
{
  return std::isfinite(sample) && sample >= kMinNormalizedSample && sample <= kMaxNormalizedSample;
}

ChannelConversionResult convertMonoToStereoDuplicate(
  const std::vector<uint8_t> & input_bytes,
  const size_t frame_count)
{
  std::vector<uint8_t> output_bytes;
  output_bytes.reserve(frame_count * 2U * sizeof(float));

  for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
    const float sample = readFloat32Le(input_bytes, frame_index);
    if (!std::isfinite(sample)) {
      return ChannelConversionResult{ProcessStatus::kNonFiniteFloat32Input, frame_index, {}};
    }
    if (!isNormalizedFinite(sample)) {
      return ChannelConversionResult{ProcessStatus::kOutOfRangeFloat32Input, frame_index, {}};
    }
    appendFloat32Le(sample, output_bytes);
    appendFloat32Le(sample, output_bytes);
  }

  return ChannelConversionResult{ProcessStatus::kOk, frame_count, std::move(output_bytes)};
}

ChannelConversionResult convertStereoToMonoAverage(
  const std::vector<uint8_t> & input_bytes,
  const size_t frame_count)
{
  std::vector<uint8_t> output_bytes;
  output_bytes.reserve(frame_count * sizeof(float));

  for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
    const size_t sample_index = frame_index * 2U;
    const float left = readFloat32Le(input_bytes, sample_index);
    const float right = readFloat32Le(input_bytes, sample_index + 1U);
    if (!std::isfinite(left) || !std::isfinite(right)) {
      return ChannelConversionResult{ProcessStatus::kNonFiniteFloat32Input, frame_index, {}};
    }
    if (!isNormalizedFinite(left) || !isNormalizedFinite(right)) {
      return ChannelConversionResult{ProcessStatus::kOutOfRangeFloat32Input, frame_index, {}};
    }

    const float averaged = (left + right) * 0.5F;
    if (!std::isfinite(averaged)) {
      return ChannelConversionResult{ProcessStatus::kNonFiniteFloat32Input, frame_index, {}};
    }
    if (!isNormalizedFinite(averaged)) {
      return ChannelConversionResult{ProcessStatus::kOutOfRangeFloat32Input, frame_index, {}};
    }
    appendFloat32Le(averaged, output_bytes);
  }

  return ChannelConversionResult{ProcessStatus::kOk, frame_count, std::move(output_bytes)};
}

InternalFloat32LeChannelConvertBackend::InternalFloat32LeChannelConvertBackend(
  const InternalFloat32LeChannelConvertConfig & config)
: config_(config)
{
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.input_channels <= 0) {
    throw std::runtime_error("input.channels must be > 0");
  }
  if (config_.output_channels <= 0) {
    throw std::runtime_error("output.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32Le) {
    throw std::runtime_error("fa_channel_convert requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_channel_convert requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_channel_convert requires expected.layout=interleaved");
  }
  if (!isSupportedChannelConversion(
      config_.conversion_mode,
      config_.input_channels,
      config_.output_channels))
  {
    throw std::runtime_error(
      "fa_channel_convert supports only mono_to_stereo_duplicate 1->2 or "
      "stereo_to_mono_average 2->1");
  }
}

int InternalFloat32LeChannelConvertBackend::outputChannels() const
{
  return config_.output_channels;
}

FrameContractStatus InternalFloat32LeChannelConvertBackend::validateContract(
  const FrameContract & contract) const
{
  if (contract.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)) {
    return FrameContractStatus::kInvalidSampleRate;
  }
  if (contract.channels != static_cast<uint32_t>(config_.input_channels)) {
    return FrameContractStatus::kInvalidChannels;
  }
  if (contract.encoding != config_.expected_encoding) {
    return FrameContractStatus::kUnsupportedEncoding;
  }
  if (contract.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)) {
    return FrameContractStatus::kUnsupportedBitDepth;
  }
  if (contract.layout != config_.expected_layout) {
    return FrameContractStatus::kUnsupportedLayout;
  }
  if (contract.data_size == 0) {
    return FrameContractStatus::kEmptyData;
  }

  const size_t bytes_per_frame = static_cast<size_t>(config_.input_channels) * sizeof(float);
  if (bytes_per_frame == 0 || (contract.data_size % bytes_per_frame) != 0) {
    return FrameContractStatus::kUnalignedData;
  }
  return FrameContractStatus::kOk;
}

ProcessResult InternalFloat32LeChannelConvertBackend::process(
  const std::vector<uint8_t> & input,
  const FrameContract & contract,
  std::vector<uint8_t> & output) const
{
  const FrameContractStatus contract_status = validateContract(contract);
  if (contract_status != FrameContractStatus::kOk) {
    return ProcessResult{ProcessStatus::kInvalidFrameContract, contract_status, 0};
  }

  const size_t frame_count = input.size() /
    (static_cast<size_t>(config_.input_channels) * sizeof(float));
  ChannelConversionResult conversion;
  if (config_.conversion_mode == kModeMonoToStereoDuplicate) {
    conversion = convertMonoToStereoDuplicate(input, frame_count);
  } else if (config_.conversion_mode == kModeStereoToMonoAverage) {
    conversion = convertStereoToMonoAverage(input, frame_count);
  } else {
    return ProcessResult{ProcessStatus::kUnsupportedConversion, FrameContractStatus::kOk, 0};
  }

  if (conversion.status != ProcessStatus::kOk || conversion.data.empty()) {
    return ProcessResult{conversion.status, FrameContractStatus::kOk, conversion.frames};
  }

  output = std::move(conversion.data);
  return ProcessResult{ProcessStatus::kOk, FrameContractStatus::kOk, conversion.frames};
}

}  // namespace fa_channel_convert::backends

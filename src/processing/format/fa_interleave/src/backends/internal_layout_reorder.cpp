#include "fa_interleave/backends/internal_layout_reorder.hpp"

#include <stdexcept>
#include <utility>

namespace fa_interleave::backends
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
    case FrameContractStatus::kUnsupportedInputLayout:
      return "unsupported_input_layout";
    case FrameContractStatus::kUnsupportedEncoding:
      return "unsupported_encoding";
    case FrameContractStatus::kUnsupportedBitDepth:
      return "unsupported_bit_depth";
    case FrameContractStatus::kEmptyData:
      return "empty_data";
    case FrameContractStatus::kUnalignedData:
      return "unaligned_data";
  }
  throw std::logic_error("unhandled layout reorder frame contract status");
}

const char * processStatusMessage(const ProcessStatus status)
{
  switch (status) {
    case ProcessStatus::kOk:
      return "ok";
    case ProcessStatus::kInvalidFrameContract:
      return "invalid layout reorder frame contract";
    case ProcessStatus::kReorderFailed:
      return "layout reorder failed";
  }
  throw std::logic_error("unhandled layout reorder backend process status");
}

bool isSupportedLayout(const std::string & layout)
{
  return layout == kInterleavedLayout || layout == kPlanarLayout;
}

bool isSupportedLayoutConversion(
  const std::string & input_layout,
  const std::string & output_layout)
{
  return isSupportedLayout(input_layout) && isSupportedLayout(output_layout) &&
         input_layout != output_layout;
}

bool isSupportedFormat(const std::string & encoding, const int bit_depth)
{
  return (encoding == kEncodingFloat32Le && bit_depth == 32) ||
         (encoding == kEncodingPcm16Le && bit_depth == 16) ||
         (encoding == kEncodingPcm32Le && bit_depth == 32);
}

size_t bytesPerSample(const std::string & encoding, const int bit_depth)
{
  if (encoding == kEncodingPcm16Le && bit_depth == 16) {
    return sizeof(uint16_t);
  }
  if ((encoding == kEncodingPcm32Le && bit_depth == 32) ||
      (encoding == kEncodingFloat32Le && bit_depth == 32))
  {
    return sizeof(uint32_t);
  }
  return 0;
}

void appendSampleBytes(
  const std::vector<uint8_t> & input_bytes,
  const size_t sample_index,
  const size_t bytes_per_sample,
  std::vector<uint8_t> & output_bytes)
{
  const size_t byte_offset = sample_index * bytes_per_sample;
  for (size_t byte_index = 0; byte_index < bytes_per_sample; ++byte_index) {
    output_bytes.push_back(input_bytes.at(byte_offset + byte_index));
  }
}

std::vector<uint8_t> reorderInterleavedToPlanar(
  const std::vector<uint8_t> & input_bytes,
  const size_t frame_count,
  const size_t channel_count,
  const size_t bytes_per_sample)
{
  std::vector<uint8_t> output_bytes;
  output_bytes.reserve(input_bytes.size());

  for (size_t channel_index = 0; channel_index < channel_count; ++channel_index) {
    for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
      const size_t input_sample_index = frame_index * channel_count + channel_index;
      appendSampleBytes(input_bytes, input_sample_index, bytes_per_sample, output_bytes);
    }
  }
  return output_bytes;
}

std::vector<uint8_t> reorderPlanarToInterleaved(
  const std::vector<uint8_t> & input_bytes,
  const size_t frame_count,
  const size_t channel_count,
  const size_t bytes_per_sample)
{
  std::vector<uint8_t> output_bytes;
  output_bytes.reserve(input_bytes.size());

  for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
    for (size_t channel_index = 0; channel_index < channel_count; ++channel_index) {
      const size_t input_sample_index = channel_index * frame_count + frame_index;
      appendSampleBytes(input_bytes, input_sample_index, bytes_per_sample, output_bytes);
    }
  }
  return output_bytes;
}

InternalLayoutReorderBackend::InternalLayoutReorderBackend(
  const InternalLayoutReorderConfig & config)
: config_(config)
{
  if (!isSupportedLayoutConversion(config_.input_layout, config_.output_layout)) {
    throw std::runtime_error(
      "fa_interleave requires interleaved->planar or planar->interleaved layout conversion");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (!isSupportedFormat(config_.expected_encoding, config_.expected_bit_depth)) {
    throw std::runtime_error("fa_interleave supports only FLOAT32LE/32, PCM16LE/16, or PCM32LE/32");
  }
}

const std::string & InternalLayoutReorderBackend::outputLayout() const
{
  return config_.output_layout;
}

FrameContractStatus InternalLayoutReorderBackend::validateContract(
  const FrameContract & contract) const
{
  if (contract.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)) {
    return FrameContractStatus::kInvalidSampleRate;
  }
  if (contract.channels != static_cast<uint32_t>(config_.expected_channels)) {
    return FrameContractStatus::kInvalidChannels;
  }
  if (contract.layout != config_.input_layout) {
    return FrameContractStatus::kUnsupportedInputLayout;
  }
  if (contract.encoding != config_.expected_encoding) {
    return FrameContractStatus::kUnsupportedEncoding;
  }
  if (contract.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)) {
    return FrameContractStatus::kUnsupportedBitDepth;
  }
  if (contract.data_size == 0) {
    return FrameContractStatus::kEmptyData;
  }

  const size_t bytes_per_sample =
    bytesPerSample(config_.expected_encoding, config_.expected_bit_depth);
  const size_t bytes_per_frame =
    static_cast<size_t>(config_.expected_channels) * bytes_per_sample;
  if (bytes_per_sample == 0 || bytes_per_frame == 0 ||
      (contract.data_size % bytes_per_frame) != 0)
  {
    return FrameContractStatus::kUnalignedData;
  }
  return FrameContractStatus::kOk;
}

ProcessResult InternalLayoutReorderBackend::process(
  const std::vector<uint8_t> & input,
  const FrameContract & contract,
  std::vector<uint8_t> & output) const
{
  const FrameContractStatus contract_status = validateContract(contract);
  if (contract_status != FrameContractStatus::kOk) {
    return ProcessResult{ProcessStatus::kInvalidFrameContract, contract_status, 0};
  }

  const size_t bytes_per_sample =
    bytesPerSample(config_.expected_encoding, config_.expected_bit_depth);
  const size_t channel_count = static_cast<size_t>(config_.expected_channels);
  const size_t frame_count = input.size() / (channel_count * bytes_per_sample);

  std::vector<uint8_t> next_output;
  if (config_.input_layout == kInterleavedLayout && config_.output_layout == kPlanarLayout) {
    next_output = reorderInterleavedToPlanar(input, frame_count, channel_count, bytes_per_sample);
  } else if (config_.input_layout == kPlanarLayout && config_.output_layout == kInterleavedLayout) {
    next_output = reorderPlanarToInterleaved(input, frame_count, channel_count, bytes_per_sample);
  } else {
    return ProcessResult{ProcessStatus::kInvalidFrameContract, FrameContractStatus::kOk, 0};
  }

  if (next_output.size() != input.size()) {
    return ProcessResult{ProcessStatus::kReorderFailed, FrameContractStatus::kOk, 0};
  }

  output = std::move(next_output);
  return ProcessResult{ProcessStatus::kOk, FrameContractStatus::kOk, frame_count};
}

}  // namespace fa_interleave::backends

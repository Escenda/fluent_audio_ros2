#include "fa_bit_depth/backends/internal_integer_bit_depth.hpp"

#include <stdexcept>
#include <utility>

namespace fa_bit_depth::backends
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
    case FrameContractStatus::kUnsupportedInputEncoding:
      return "unsupported_input_encoding";
    case FrameContractStatus::kUnsupportedInputBitDepth:
      return "unsupported_input_bit_depth";
    case FrameContractStatus::kUnsupportedOutputEncoding:
      return "unsupported_output_encoding";
    case FrameContractStatus::kUnsupportedOutputBitDepth:
      return "unsupported_output_bit_depth";
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
      return "invalid PCM integer bit-depth frame contract";
    case ProcessStatus::kConversionFailed:
      return "PCM integer bit-depth conversion failed";
  }
  return "unknown bit-depth backend status";
}

bool isSupportedConversion(
  const std::string & input_encoding,
  const int input_bit_depth,
  const std::string & output_encoding,
  const int output_bit_depth)
{
  return input_encoding == kEncodingPcm16Le && input_bit_depth == 16 &&
         output_encoding == kEncodingPcm32Le && output_bit_depth == 32;
}

size_t bytesPerSample(const int bit_depth)
{
  if (bit_depth <= 0 || (bit_depth % 8) != 0) {
    return 0;
  }
  return static_cast<size_t>(bit_depth / 8);
}

void appendPcm32Le(const uint32_t sample, std::vector<uint8_t> & out_bytes)
{
  out_bytes.push_back(static_cast<uint8_t>(sample & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((sample >> 8U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((sample >> 16U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((sample >> 24U) & 0xFFU));
}

std::vector<uint8_t> convertPcm16ToPcm32(const std::vector<uint8_t> & input_bytes)
{
  if (input_bytes.empty() || (input_bytes.size() % sizeof(uint16_t)) != 0) {
    return {};
  }

  std::vector<uint8_t> out_bytes;
  out_bytes.reserve((input_bytes.size() / sizeof(uint16_t)) * sizeof(uint32_t));
  for (size_t i = 0; i < input_bytes.size(); i += sizeof(uint16_t)) {
    const uint16_t raw =
      static_cast<uint16_t>(input_bytes.at(i)) |
      (static_cast<uint16_t>(input_bytes.at(i + 1)) << 8U);
    const uint32_t aligned_sample = static_cast<uint32_t>(raw) << 16U;
    appendPcm32Le(aligned_sample, out_bytes);
  }
  return out_bytes;
}

InternalIntegerBitDepthBackend::InternalIntegerBitDepthBackend(
  const InternalIntegerBitDepthConfig & config)
: config_(config)
{
  if (!isSupportedConversion(
      config_.input_encoding,
      config_.input_bit_depth,
      config_.output_encoding,
      config_.output_bit_depth))
  {
    throw std::runtime_error(
      "fa_bit_depth supports only lossless PCM16LE/16 -> PCM32LE/32 integer bit-depth expansion");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_bit_depth requires expected.layout=interleaved");
  }
}

const std::string & InternalIntegerBitDepthBackend::outputEncoding() const
{
  return config_.output_encoding;
}

int InternalIntegerBitDepthBackend::outputBitDepth() const
{
  return config_.output_bit_depth;
}

FrameContractStatus InternalIntegerBitDepthBackend::validateContract(
  const FrameContract & contract) const
{
  if (contract.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)) {
    return FrameContractStatus::kInvalidSampleRate;
  }
  if (contract.channels != static_cast<uint32_t>(config_.expected_channels)) {
    return FrameContractStatus::kInvalidChannels;
  }
  if (contract.input_encoding != config_.input_encoding) {
    return FrameContractStatus::kUnsupportedInputEncoding;
  }
  if (contract.input_bit_depth != static_cast<uint32_t>(config_.input_bit_depth)) {
    return FrameContractStatus::kUnsupportedInputBitDepth;
  }
  if (config_.output_encoding != kEncodingPcm32Le) {
    return FrameContractStatus::kUnsupportedOutputEncoding;
  }
  if (config_.output_bit_depth != 32) {
    return FrameContractStatus::kUnsupportedOutputBitDepth;
  }
  if (contract.layout != config_.expected_layout) {
    return FrameContractStatus::kUnsupportedLayout;
  }
  if (contract.data_size == 0) {
    return FrameContractStatus::kEmptyData;
  }

  const size_t input_bytes_per_sample = bytesPerSample(config_.input_bit_depth);
  const size_t bytes_per_frame =
    static_cast<size_t>(config_.expected_channels) * input_bytes_per_sample;
  if (input_bytes_per_sample == 0 || bytes_per_frame == 0 ||
      (contract.data_size % bytes_per_frame) != 0)
  {
    return FrameContractStatus::kUnalignedData;
  }
  return FrameContractStatus::kOk;
}

ProcessResult InternalIntegerBitDepthBackend::process(
  const std::vector<uint8_t> & input,
  const FrameContract & contract,
  std::vector<uint8_t> & output) const
{
  const FrameContractStatus contract_status = validateContract(contract);
  if (contract_status != FrameContractStatus::kOk) {
    return ProcessResult{ProcessStatus::kInvalidFrameContract, contract_status, 0};
  }

  std::vector<uint8_t> next_output = convertPcm16ToPcm32(input);
  if (next_output.empty()) {
    return ProcessResult{ProcessStatus::kConversionFailed, FrameContractStatus::kOk, 0};
  }

  output = std::move(next_output);
  return ProcessResult{
    ProcessStatus::kOk,
    FrameContractStatus::kOk,
    input.size() / sizeof(uint16_t)};
}

}  // namespace fa_bit_depth::backends

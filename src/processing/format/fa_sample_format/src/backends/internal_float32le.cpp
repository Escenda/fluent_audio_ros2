#include "fa_sample_format/backends/internal_float32le.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fa_sample_format::backends
{

namespace
{
constexpr float kPcm16Scale = 32768.0F;
constexpr double kPcm32Scale = 2147483648.0;

bool isFiniteNormalizedFloat32(const float sample)
{
  return std::isfinite(sample) && sample >= -1.0F && sample <= 1.0F;
}
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
  throw std::logic_error("unhandled sample-format frame contract status");
}

const char * processStatusMessage(const ProcessStatus status)
{
  switch (status) {
    case ProcessStatus::kOk:
      return "ok";
    case ProcessStatus::kInvalidFrameContract:
      return "invalid sample-format frame contract";
    case ProcessStatus::kNonFiniteFloat32Input:
      return "FLOAT32LE input sample is not finite";
    case ProcessStatus::kOutOfRangeFloat32Input:
      return "FLOAT32LE input sample is outside normalized range";
    case ProcessStatus::kConversionFailed:
      return "sample-format conversion failed";
  }
  throw std::logic_error("unhandled sample-format backend process status");
}

bool isSupportedSampleFormatConversion(
  const std::string & input_encoding,
  const int input_bit_depth,
  const std::string & output_encoding,
  const int output_bit_depth)
{
  return (
           output_encoding == kEncodingFloat32Le && output_bit_depth == 32 &&
           (
             (input_encoding == kEncodingPcm16Le && input_bit_depth == 16) ||
             (input_encoding == kEncodingPcm32Le && input_bit_depth == 32)
           )
         ) ||
         (
           input_encoding == kEncodingFloat32Le && input_bit_depth == 32 &&
           output_encoding == kEncodingPcm16Le && output_bit_depth == 16
         );
}

size_t bytesPerSample(const int bit_depth)
{
  if (bit_depth <= 0 || (bit_depth % 8) != 0) {
    return 0;
  }
  return static_cast<size_t>(bit_depth / 8);
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

void appendPcm16Le(const int16_t sample, std::vector<uint8_t> & out_bytes)
{
  const uint16_t raw = static_cast<uint16_t>(sample);
  out_bytes.push_back(static_cast<uint8_t>(raw & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 8U) & 0xFFU));
}

ByteConversionResult convertPcm16ToFloat32(const std::vector<uint8_t> & input_bytes)
{
  if (input_bytes.empty() || (input_bytes.size() % sizeof(uint16_t)) != 0) {
    return ByteConversionResult{ProcessStatus::kConversionFailed, 0, {}};
  }

  std::vector<uint8_t> out_bytes;
  out_bytes.reserve((input_bytes.size() / sizeof(uint16_t)) * sizeof(float));
  for (size_t i = 0; i < input_bytes.size(); i += sizeof(uint16_t)) {
    const uint16_t raw =
      static_cast<uint16_t>(input_bytes.at(i)) |
      (static_cast<uint16_t>(input_bytes.at(i + 1)) << 8U);
    const int32_t signed_value = raw >= 0x8000U ?
      static_cast<int32_t>(raw) - 0x10000 :
      static_cast<int32_t>(raw);
    appendFloat32Le(static_cast<float>(signed_value) / kPcm16Scale, out_bytes);
  }

  return ByteConversionResult{
    ProcessStatus::kOk,
    input_bytes.size() / sizeof(uint16_t),
    std::move(out_bytes)};
}

ByteConversionResult convertPcm32ToFloat32(const std::vector<uint8_t> & input_bytes)
{
  if (input_bytes.empty() || (input_bytes.size() % sizeof(uint32_t)) != 0) {
    return ByteConversionResult{ProcessStatus::kConversionFailed, 0, {}};
  }

  std::vector<uint8_t> out_bytes;
  out_bytes.reserve((input_bytes.size() / sizeof(uint32_t)) * sizeof(float));
  for (size_t i = 0; i < input_bytes.size(); i += sizeof(uint32_t)) {
    const uint32_t raw =
      static_cast<uint32_t>(input_bytes.at(i)) |
      (static_cast<uint32_t>(input_bytes.at(i + 1)) << 8U) |
      (static_cast<uint32_t>(input_bytes.at(i + 2)) << 16U) |
      (static_cast<uint32_t>(input_bytes.at(i + 3)) << 24U);
    const int64_t signed_value = raw >= 0x80000000UL ?
      static_cast<int64_t>(raw) - 0x100000000LL :
      static_cast<int64_t>(raw);
    appendFloat32Le(static_cast<float>(static_cast<double>(signed_value) / kPcm32Scale), out_bytes);
  }

  return ByteConversionResult{
    ProcessStatus::kOk,
    input_bytes.size() / sizeof(uint32_t),
    std::move(out_bytes)};
}

ByteConversionResult convertFloat32ToPcm16(const std::vector<uint8_t> & input_bytes)
{
  if (input_bytes.empty() || (input_bytes.size() % sizeof(float)) != 0) {
    return ByteConversionResult{ProcessStatus::kConversionFailed, 0, {}};
  }

  std::vector<uint8_t> out_bytes;
  out_bytes.reserve((input_bytes.size() / sizeof(float)) * sizeof(int16_t));
  for (size_t i = 0; i < input_bytes.size(); i += sizeof(float)) {
    const uint32_t raw =
      static_cast<uint32_t>(input_bytes.at(i)) |
      (static_cast<uint32_t>(input_bytes.at(i + 1)) << 8U) |
      (static_cast<uint32_t>(input_bytes.at(i + 2)) << 16U) |
      (static_cast<uint32_t>(input_bytes.at(i + 3)) << 24U);
    float sample = 0.0F;
    std::memcpy(&sample, &raw, sizeof(float));
    if (!std::isfinite(sample)) {
      return ByteConversionResult{ProcessStatus::kNonFiniteFloat32Input, 0, {}};
    }
    if (!isFiniteNormalizedFloat32(sample)) {
      return ByteConversionResult{ProcessStatus::kOutOfRangeFloat32Input, 0, {}};
    }

    const double scaled = sample < 0.0F ?
      static_cast<double>(sample) * 32768.0 :
      static_cast<double>(sample) * 32767.0;
    const int32_t rounded = static_cast<int32_t>(std::lround(scaled));
    if (rounded < -32768 || rounded > 32767) {
      return ByteConversionResult{ProcessStatus::kOutOfRangeFloat32Input, 0, {}};
    }
    appendPcm16Le(static_cast<int16_t>(rounded), out_bytes);
  }

  return ByteConversionResult{
    ProcessStatus::kOk,
    input_bytes.size() / sizeof(float),
    std::move(out_bytes)};
}

InternalFloat32LeBackend::InternalFloat32LeBackend(const InternalFloat32LeConfig & config)
: config_(config)
{
  if (!isSupportedSampleFormatConversion(
      config_.input_encoding,
      config_.input_bit_depth,
      config_.output_encoding,
      config_.output_bit_depth))
  {
    throw std::runtime_error(
      "fa_sample_format supports only PCM16LE/16 or PCM32LE/32 to FLOAT32LE/32, or "
      "FLOAT32LE/32 to PCM16LE/16");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_sample_format requires expected.layout=interleaved");
  }
}

const std::string & InternalFloat32LeBackend::outputEncoding() const
{
  return config_.output_encoding;
}

int InternalFloat32LeBackend::outputBitDepth() const
{
  return config_.output_bit_depth;
}

FrameContractStatus InternalFloat32LeBackend::validateContract(const FrameContract & contract) const
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
  if (config_.output_encoding != kEncodingFloat32Le &&
      config_.output_encoding != kEncodingPcm16Le)
  {
    return FrameContractStatus::kUnsupportedOutputEncoding;
  }
  if (config_.output_bit_depth != 32 && config_.output_bit_depth != 16) {
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

ProcessResult InternalFloat32LeBackend::process(
  const std::vector<uint8_t> & input,
  const FrameContract & contract,
  std::vector<uint8_t> & output) const
{
  const FrameContractStatus contract_status = validateContract(contract);
  if (contract_status != FrameContractStatus::kOk) {
    return ProcessResult{ProcessStatus::kInvalidFrameContract, contract_status, 0};
  }

  ByteConversionResult conversion;
  if (config_.input_encoding == kEncodingPcm16Le && config_.input_bit_depth == 16) {
    conversion = convertPcm16ToFloat32(input);
  } else if (config_.input_encoding == kEncodingPcm32Le && config_.input_bit_depth == 32) {
    conversion = convertPcm32ToFloat32(input);
  } else if (config_.input_encoding == kEncodingFloat32Le && config_.input_bit_depth == 32 &&
             config_.output_encoding == kEncodingPcm16Le && config_.output_bit_depth == 16)
  {
    conversion = convertFloat32ToPcm16(input);
  } else {
    return ProcessResult{ProcessStatus::kInvalidFrameContract, FrameContractStatus::kOk, 0};
  }

  if (conversion.status != ProcessStatus::kOk || conversion.data.empty()) {
    return ProcessResult{conversion.status, FrameContractStatus::kOk, conversion.samples};
  }

  output = std::move(conversion.data);
  return ProcessResult{ProcessStatus::kOk, FrameContractStatus::kOk, conversion.samples};
}

}  // namespace fa_sample_format::backends

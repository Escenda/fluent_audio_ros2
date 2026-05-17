#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace fa_sample_format
{

inline constexpr const char * kEncodingPcm16 = "PCM16LE";
inline constexpr const char * kEncodingPcm32 = "PCM32LE";
inline constexpr const char * kEncodingFloat32 = "FLOAT32LE";
inline constexpr float kPcm16Scale = 32768.0F;
inline constexpr double kPcm32Scale = 2147483648.0;

inline bool isSupportedSampleFormatConversion(
  const std::string & input_encoding,
  const int input_bit_depth,
  const std::string & output_encoding,
  const int output_bit_depth)
{
  return (
           output_encoding == kEncodingFloat32 && output_bit_depth == 32 &&
           (
             (input_encoding == kEncodingPcm16 && input_bit_depth == 16) ||
             (input_encoding == kEncodingPcm32 && input_bit_depth == 32)
           )
         ) ||
         (
           input_encoding == kEncodingFloat32 && input_bit_depth == 32 &&
           output_encoding == kEncodingPcm16 && output_bit_depth == 16
         );
}

inline void appendFloat32Le(const float sample, std::vector<uint8_t> & out_bytes)
{
  uint32_t raw = 0;
  std::memcpy(&raw, &sample, sizeof(float));
  out_bytes.push_back(static_cast<uint8_t>(raw & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 8U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 16U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 24U) & 0xFFU));
}

inline std::vector<uint8_t> convertPcm16ToFloat32(const std::vector<uint8_t> & input_bytes)
{
  if (input_bytes.empty() || (input_bytes.size() % sizeof(uint16_t)) != 0) {
    return {};
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
  return out_bytes;
}

inline std::vector<uint8_t> convertPcm32ToFloat32(const std::vector<uint8_t> & input_bytes)
{
  if (input_bytes.empty() || (input_bytes.size() % sizeof(uint32_t)) != 0) {
    return {};
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
  return out_bytes;
}

inline void appendPcm16Le(const int16_t sample, std::vector<uint8_t> & out_bytes)
{
  const uint16_t raw = static_cast<uint16_t>(sample);
  out_bytes.push_back(static_cast<uint8_t>(raw & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 8U) & 0xFFU));
}

inline std::vector<uint8_t> convertFloat32ToPcm16(const std::vector<uint8_t> & input_bytes)
{
  if (input_bytes.empty() || (input_bytes.size() % sizeof(float)) != 0) {
    return {};
  }

  std::vector<uint8_t> out_bytes;
  out_bytes.reserve((input_bytes.size() / sizeof(float)) * sizeof(int16_t));
  for (size_t i = 0; i < input_bytes.size(); i += sizeof(float)) {
    float sample = 0.0F;
    std::memcpy(&sample, input_bytes.data() + i, sizeof(float));
    if (!std::isfinite(sample) || sample < -1.0F || sample > 1.0F) {
      return {};
    }
    const double scaled = sample < 0.0F ?
      static_cast<double>(sample) * 32768.0 :
      static_cast<double>(sample) * 32767.0;
    const int32_t rounded = static_cast<int32_t>(std::lround(scaled));
    if (rounded < -32768 || rounded > 32767) {
      return {};
    }
    appendPcm16Le(static_cast<int16_t>(rounded), out_bytes);
  }
  return out_bytes;
}

}  // namespace fa_sample_format

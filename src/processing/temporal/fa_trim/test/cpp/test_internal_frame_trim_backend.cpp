#include "fa_trim/backends/internal_frame_trim.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace
{

std::vector<uint8_t> float32LeBytes(const std::vector<float> & samples)
{
  std::vector<uint8_t> bytes;
  bytes.resize(samples.size() * sizeof(float));
  for (size_t i = 0; i < samples.size(); ++i) {
    uint32_t bits = 0U;
    std::memcpy(&bits, &samples[i], sizeof(bits));
    const size_t offset = i * sizeof(float);
    bytes[offset] = static_cast<uint8_t>(bits & 0xFFU);
    bytes[offset + 1U] = static_cast<uint8_t>((bits >> 8U) & 0xFFU);
    bytes[offset + 2U] = static_cast<uint8_t>((bits >> 16U) & 0xFFU);
    bytes[offset + 3U] = static_cast<uint8_t>((bits >> 24U) & 0xFFU);
  }
  return bytes;
}

std::vector<float> decodeFloat32Le(const std::vector<uint8_t> & bytes)
{
  std::vector<float> samples;
  samples.reserve(bytes.size() / sizeof(float));
  for (size_t offset = 0; offset < bytes.size(); offset += sizeof(float)) {
    const uint32_t bits =
      static_cast<uint32_t>(bytes[offset]) |
      (static_cast<uint32_t>(bytes[offset + 1U]) << 8U) |
      (static_cast<uint32_t>(bytes[offset + 2U]) << 16U) |
      (static_cast<uint32_t>(bytes[offset + 3U]) << 24U);
    float sample = 0.0F;
    std::memcpy(&sample, &bits, sizeof(sample));
    samples.push_back(sample);
  }
  return samples;
}

}  // namespace

TEST(InternalFrameTrimBackendContract, TrimsLeadingAndTrailingFrames)
{
  fa_trim::backends::InternalFrameTrimBackend backend(
    fa_trim::backends::InternalFrameTrimConfig{1, 1U, 1U});

  std::vector<uint8_t> output;
  const auto result = backend.process(float32LeBytes({0.1F, 0.2F, 0.3F, 0.4F}), output);

  ASSERT_EQ(result.status, fa_trim::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.input_frame_count, 4U);
  EXPECT_EQ(result.output_frame_count, 2U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.2F, 0.3F}));
}

TEST(InternalFrameTrimBackendContract, KeepsInterleavedChannelFrames)
{
  fa_trim::backends::InternalFrameTrimBackend backend(
    fa_trim::backends::InternalFrameTrimConfig{2, 1U, 0U});

  std::vector<uint8_t> output;
  const auto result = backend.process(
    float32LeBytes({0.1F, -0.1F, 0.2F, -0.2F, 0.3F, -0.3F}),
    output);

  ASSERT_EQ(result.status, fa_trim::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.input_frame_count, 3U);
  EXPECT_EQ(result.output_frame_count, 2U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.2F, -0.2F, 0.3F, -0.3F}));
}

TEST(InternalFrameTrimBackendContract, ReportsInputRejectionStatuses)
{
  fa_trim::backends::InternalFrameTrimBackend backend(
    fa_trim::backends::InternalFrameTrimConfig{1, 1U, 0U});

  std::vector<uint8_t> output = float32LeBytes({0.5F});
  EXPECT_EQ(
    backend.process({}, output).status,
    fa_trim::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(output, float32LeBytes({0.5F}));

  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0U, 1U, 2U}, output).status,
    fa_trim::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_trim::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output).status,
    fa_trim::backends::ProcessStatus::kOutOfRangeInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({0.0F}), output).status,
    fa_trim::backends::ProcessStatus::kTrimExhaustsInput);
}

TEST(InternalFrameTrimBackendContract, RejectsInvalidConfiguration)
{
  EXPECT_THROW(
    fa_trim::backends::InternalFrameTrimBackend(
      fa_trim::backends::InternalFrameTrimConfig{0, 1U, 0U}),
    std::runtime_error);
  EXPECT_THROW(
    fa_trim::backends::InternalFrameTrimBackend(
      fa_trim::backends::InternalFrameTrimConfig{1, 0U, 0U}),
    std::runtime_error);
}

TEST(InternalFrameTrimBackendContract, FailsClosedForUnhandledStatusMessage)
{
  EXPECT_THROW(
    static_cast<void>(
      fa_trim::backends::processStatusMessage(
        static_cast<fa_trim::backends::ProcessStatus>(999))),
    std::logic_error);
}

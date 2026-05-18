#include "fa_delay/backends/internal_sample_delay.hpp"

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

TEST(InternalSampleDelayBackendContract, AppliesDelayAcrossFrames)
{
  fa_delay::backends::InternalSampleDelayBackend backend(
    fa_delay::backends::InternalSampleDelayConfig{1, 2U});

  std::vector<uint8_t> output;
  auto first = backend.process("mic", float32LeBytes({1.0F, 0.5F}), output);
  ASSERT_EQ(first.status, fa_delay::backends::ProcessStatus::kOk);
  EXPECT_FALSE(first.source_reset);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.0F, 0.0F}));

  auto second = backend.process("mic", float32LeBytes({-0.5F}), output);
  ASSERT_EQ(second.status, fa_delay::backends::ProcessStatus::kOk);
  EXPECT_FALSE(second.source_reset);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{1.0F}));
}

TEST(InternalSampleDelayBackendContract, KeepsChannelBuffersIndependent)
{
  fa_delay::backends::InternalSampleDelayBackend backend(
    fa_delay::backends::InternalSampleDelayConfig{2, 1U});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process("mic", float32LeBytes({0.25F, -0.25F}), output).status,
    fa_delay::backends::ProcessStatus::kOk);
  ASSERT_EQ(
    backend.process("mic", float32LeBytes({0.0F, 0.0F}), output).status,
    fa_delay::backends::ProcessStatus::kOk);

  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.25F, -0.25F}));
}

TEST(InternalSampleDelayBackendContract, ResetsStateOnSourceChange)
{
  fa_delay::backends::InternalSampleDelayBackend backend(
    fa_delay::backends::InternalSampleDelayConfig{1, 1U});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process("mic-a", float32LeBytes({1.0F}), output).status,
    fa_delay::backends::ProcessStatus::kOk);

  const auto result = backend.process("mic-b", float32LeBytes({0.0F}), output);
  ASSERT_EQ(result.status, fa_delay::backends::ProcessStatus::kOk);
  EXPECT_TRUE(result.source_reset);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.0F}));
  EXPECT_EQ(backend.currentSourceId(), "mic-b");
}

TEST(InternalSampleDelayBackendContract, ReportsInputRejectionStatuses)
{
  fa_delay::backends::InternalSampleDelayBackend backend(
    fa_delay::backends::InternalSampleDelayConfig{1, 1U});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process("", float32LeBytes({0.0F}), output).status,
    fa_delay::backends::ProcessStatus::kEmptySourceId);
  EXPECT_EQ(
    backend.process("mic", {}, output).status,
    fa_delay::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process("mic", std::vector<uint8_t>{0U, 1U, 2U}, output).status,
    fa_delay::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process("mic", float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_delay::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process("mic", float32LeBytes({1.25F}), output).status,
    fa_delay::backends::ProcessStatus::kOutOfRangeInput);

  EXPECT_TRUE(backend.currentSourceId().empty());
}

TEST(InternalSampleDelayBackendContract, RejectsInvalidConfiguration)
{
  EXPECT_THROW(
    fa_delay::backends::InternalSampleDelayBackend(
      fa_delay::backends::InternalSampleDelayConfig{0, 1U}),
    std::runtime_error);
  EXPECT_THROW(
    fa_delay::backends::InternalSampleDelayBackend(
      fa_delay::backends::InternalSampleDelayConfig{1, 0U}),
    std::runtime_error);
}

TEST(InternalSampleDelayBackendContract, FailsClosedForUnhandledStatusMessage)
{
  EXPECT_THROW(
    static_cast<void>(
      fa_delay::backends::processStatusMessage(
        static_cast<fa_delay::backends::ProcessStatus>(999))),
    std::logic_error);
}

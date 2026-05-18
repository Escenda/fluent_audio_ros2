#include "fa_reverb/backends/internal_feedback_delay.hpp"

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

TEST(InternalFeedbackDelayBackendContract, AppliesDeterministicFeedbackDelay)
{
  fa_reverb::backends::InternalFeedbackDelayBackend backend(
    fa_reverb::backends::InternalFeedbackDelayConfig{100, 1, 0.0, 0.0, 0.5, 0.5});

  std::vector<uint8_t> output;
  const auto result = backend.process("mic", float32LeBytes({1.0F, 0.0F, 0.0F, 0.0F}), output);

  ASSERT_EQ(result.status, fa_reverb::backends::ProcessStatus::kOk);
  EXPECT_FALSE(result.source_reset);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.5F, 0.0F, 0.0F, 0.1F}));
}

TEST(InternalFeedbackDelayBackendContract, KeepsChannelDelayLinesIndependent)
{
  fa_reverb::backends::InternalFeedbackDelayBackend backend(
    fa_reverb::backends::InternalFeedbackDelayConfig{100, 2, 0.0, 0.0, 0.5, 0.5});

  std::vector<uint8_t> output;
  const auto result = backend.process(
    "mic",
    float32LeBytes({1.0F, -1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F}),
    output);

  ASSERT_EQ(result.status, fa_reverb::backends::ProcessStatus::kOk);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.5F, -0.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, -0.1F}));
}

TEST(InternalFeedbackDelayBackendContract, ResetsStateOnSourceChange)
{
  fa_reverb::backends::InternalFeedbackDelayBackend backend(
    fa_reverb::backends::InternalFeedbackDelayConfig{100, 1, 0.0, 0.0, 0.5, 0.5});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process("mic-a", float32LeBytes({1.0F}), output).status,
    fa_reverb::backends::ProcessStatus::kOk);

  const auto result = backend.process("mic-b", float32LeBytes({0.0F, 0.0F, 0.0F, 0.0F}), output);
  ASSERT_EQ(result.status, fa_reverb::backends::ProcessStatus::kOk);
  EXPECT_TRUE(result.source_reset);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F}));
  EXPECT_EQ(backend.currentSourceId(), "mic-b");
}

TEST(InternalFeedbackDelayBackendContract, ReportsInputRejectionStatuses)
{
  fa_reverb::backends::InternalFeedbackDelayBackend backend(
    fa_reverb::backends::InternalFeedbackDelayConfig{100, 1, 0.0, 0.0, 0.5, 0.5});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process("", float32LeBytes({0.0F}), output).status,
    fa_reverb::backends::ProcessStatus::kEmptySourceId);
  EXPECT_EQ(
    backend.process("mic", {}, output).status,
    fa_reverb::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process("mic", std::vector<uint8_t>{0U, 1U, 2U}, output).status,
    fa_reverb::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process("mic", float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_reverb::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process("mic", float32LeBytes({1.5F}), output).status,
    fa_reverb::backends::ProcessStatus::kOutOfRangeInput);

  EXPECT_TRUE(backend.currentSourceId().empty());
}

TEST(InternalFeedbackDelayBackendContract, RejectsInvalidConfiguration)
{
  EXPECT_THROW(
    fa_reverb::backends::InternalFeedbackDelayBackend(
      (fa_reverb::backends::InternalFeedbackDelayConfig{100, 1, 0.0, 0.0, 0.75, 0.75})),
    std::runtime_error);
  EXPECT_THROW(
    fa_reverb::backends::InternalFeedbackDelayBackend(
      (fa_reverb::backends::InternalFeedbackDelayConfig{0, 1, 0.0, 0.0, 0.5, 0.5})),
    std::runtime_error);
}

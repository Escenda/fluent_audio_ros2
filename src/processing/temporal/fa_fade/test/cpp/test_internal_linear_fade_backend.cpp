#include "fa_fade/backends/internal_linear_fade.hpp"

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

TEST(InternalLinearFadeBackendContract, AppliesFadeInAcrossFrames)
{
  fa_fade::backends::InternalLinearFadeBackend backend(
    fa_fade::backends::InternalLinearFadeConfig{
      1,
      fa_fade::backends::FadeMode::kFadeIn,
      4U,
      0U});

  std::vector<uint8_t> output;
  auto first = backend.process(float32LeBytes({1.0F, 1.0F}), output);
  ASSERT_EQ(first.status, fa_fade::backends::ProcessStatus::kOk);
  EXPECT_EQ(first.position_frames, 2U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.0F, 0.25F}));

  auto second = backend.process(float32LeBytes({1.0F, 1.0F}), output);
  ASSERT_EQ(second.status, fa_fade::backends::ProcessStatus::kOk);
  EXPECT_EQ(second.position_frames, 4U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.5F, 0.75F}));
}

TEST(InternalLinearFadeBackendContract, AppliesFadeOutAndKeepsChannelFramePosition)
{
  fa_fade::backends::InternalLinearFadeBackend backend(
    fa_fade::backends::InternalLinearFadeConfig{
      2,
      fa_fade::backends::FadeMode::kFadeOut,
      2U,
      0U});

  std::vector<uint8_t> output;
  const auto result = backend.process(float32LeBytes({1.0F, -1.0F, 1.0F, -1.0F}), output);

  ASSERT_EQ(result.status, fa_fade::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.input_frame_count, 2U);
  EXPECT_EQ(result.output_frame_count, 2U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{1.0F, -1.0F, 0.5F, -0.5F}));
}

TEST(InternalLinearFadeBackendContract, ReportsInputRejectionStatusesWithoutCommittingState)
{
  fa_fade::backends::InternalLinearFadeBackend backend(
    fa_fade::backends::InternalLinearFadeConfig{
      1,
      fa_fade::backends::FadeMode::kFadeIn,
      4U,
      0U});

  std::vector<uint8_t> output = float32LeBytes({0.5F});
  EXPECT_EQ(
    backend.process({}, output).status,
    fa_fade::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(output, float32LeBytes({0.5F}));
  EXPECT_EQ(backend.positionFrames(), 0U);

  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0U, 1U, 2U}, output).status,
    fa_fade::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_fade::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output).status,
    fa_fade::backends::ProcessStatus::kOutOfRangeInput);
  EXPECT_EQ(backend.positionFrames(), 0U);
}

TEST(InternalLinearFadeBackendContract, RejectsPositionOverflowWithoutCommittingState)
{
  fa_fade::backends::InternalLinearFadeBackend backend(
    fa_fade::backends::InternalLinearFadeConfig{
      1,
      fa_fade::backends::FadeMode::kFadeIn,
      4U,
      std::numeric_limits<uint64_t>::max()});

  std::vector<uint8_t> output;
  const auto result = backend.process(float32LeBytes({0.0F}), output);
  EXPECT_EQ(result.status, fa_fade::backends::ProcessStatus::kPositionOverflow);
  EXPECT_EQ(backend.positionFrames(), std::numeric_limits<uint64_t>::max());
}

TEST(InternalLinearFadeBackendContract, RejectsInvalidConfiguration)
{
  EXPECT_THROW(
    fa_fade::backends::InternalLinearFadeBackend(
      fa_fade::backends::InternalLinearFadeConfig{
        0,
        fa_fade::backends::FadeMode::kFadeIn,
        1U,
        0U}),
    std::runtime_error);
  EXPECT_THROW(
    fa_fade::backends::InternalLinearFadeBackend(
      fa_fade::backends::InternalLinearFadeConfig{
        1,
        fa_fade::backends::FadeMode::kFadeIn,
        0U,
        0U}),
    std::runtime_error);
  EXPECT_THROW(
    fa_fade::backends::InternalLinearFadeBackend(
      fa_fade::backends::InternalLinearFadeConfig{
        1,
        static_cast<fa_fade::backends::FadeMode>(999),
        1U,
        0U}),
    std::logic_error);
}

TEST(InternalLinearFadeBackendContract, FailsClosedForUnhandledEnumMessages)
{
  EXPECT_THROW(
    static_cast<void>(
      fa_fade::backends::fadeModeName(
        static_cast<fa_fade::backends::FadeMode>(999))),
    std::logic_error);
  EXPECT_THROW(
    static_cast<void>(
      fa_fade::backends::processStatusMessage(
        static_cast<fa_fade::backends::ProcessStatus>(999))),
    std::logic_error);
}

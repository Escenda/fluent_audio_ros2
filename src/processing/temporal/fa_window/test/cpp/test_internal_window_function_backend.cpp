#include "fa_window/backends/internal_window_function.hpp"

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

TEST(InternalWindowFunctionBackendContract, AppliesHannWindow)
{
  fa_window::backends::InternalWindowFunctionBackend backend(
    fa_window::backends::InternalWindowFunctionConfig{
      1,
      fa_window::backends::WindowType::kHann,
      3U,
      true});

  std::vector<uint8_t> output;
  const auto result = backend.process(float32LeBytes({1.0F, 1.0F, 1.0F}), output);

  ASSERT_EQ(result.status, fa_window::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.input_frame_count, 3U);
  EXPECT_EQ(result.output_frame_count, 3U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.0F, 1.0F, 0.0F}));
}

TEST(InternalWindowFunctionBackendContract, AppliesHammingWindowAndKeepsChannelFramePosition)
{
  fa_window::backends::InternalWindowFunctionBackend backend(
    fa_window::backends::InternalWindowFunctionConfig{
      2,
      fa_window::backends::WindowType::kHamming,
      3U,
      true});

  std::vector<uint8_t> output;
  const auto result = backend.process(
    float32LeBytes({1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F}),
    output);

  ASSERT_EQ(result.status, fa_window::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.input_frame_count, 3U);
  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 6U);
  EXPECT_NEAR(samples[0], 0.08F, 1.0e-6F);
  EXPECT_NEAR(samples[1], -0.08F, 1.0e-6F);
  EXPECT_NEAR(samples[2], 1.0F, 1.0e-6F);
  EXPECT_NEAR(samples[3], -1.0F, 1.0e-6F);
  EXPECT_NEAR(samples[4], 0.08F, 1.0e-6F);
  EXPECT_NEAR(samples[5], -0.08F, 1.0e-6F);
}

TEST(InternalWindowFunctionBackendContract, HandlesDynamicFrameCountWhenNotStrict)
{
  fa_window::backends::InternalWindowFunctionBackend backend(
    fa_window::backends::InternalWindowFunctionConfig{
      1,
      fa_window::backends::WindowType::kHann,
      512U,
      false});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process(float32LeBytes({1.0F}), output).status,
    fa_window::backends::ProcessStatus::kTooFewFrames);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.0F, 1.0F, 1.0F}), output).status,
    fa_window::backends::ProcessStatus::kOk);
}

TEST(InternalWindowFunctionBackendContract, ReportsInputRejectionStatusesWithoutCommittingOutput)
{
  fa_window::backends::InternalWindowFunctionBackend backend(
    fa_window::backends::InternalWindowFunctionConfig{
      1,
      fa_window::backends::WindowType::kHann,
      3U,
      true});

  std::vector<uint8_t> output = float32LeBytes({0.5F});
  EXPECT_EQ(
    backend.process({}, output).status,
    fa_window::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(output, float32LeBytes({0.5F}));
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0U, 1U, 2U}, output).status,
    fa_window::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.0F, 1.0F}), output).status,
    fa_window::backends::ProcessStatus::kFrameCountMismatch);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F}), output).status,
    fa_window::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F, 0.0F, 0.0F}), output).status,
    fa_window::backends::ProcessStatus::kOutOfRangeInput);
}

TEST(InternalWindowFunctionBackendContract, RejectsInvalidConfiguration)
{
  EXPECT_THROW(
    fa_window::backends::InternalWindowFunctionBackend(
      fa_window::backends::InternalWindowFunctionConfig{
        0,
        fa_window::backends::WindowType::kHann,
        3U,
        true}),
    std::runtime_error);
  EXPECT_THROW(
    fa_window::backends::InternalWindowFunctionBackend(
      fa_window::backends::InternalWindowFunctionConfig{
        1,
        fa_window::backends::WindowType::kHann,
        1U,
        true}),
    std::runtime_error);
  EXPECT_THROW(
    fa_window::backends::InternalWindowFunctionBackend(
      fa_window::backends::InternalWindowFunctionConfig{
        1,
        static_cast<fa_window::backends::WindowType>(999),
        3U,
        true}),
    std::logic_error);
}

TEST(InternalWindowFunctionBackendContract, FailsClosedForUnhandledEnumMessages)
{
  EXPECT_THROW(
    static_cast<void>(
      fa_window::backends::windowTypeName(
        static_cast<fa_window::backends::WindowType>(999))),
    std::logic_error);
  EXPECT_THROW(
    static_cast<void>(
      fa_window::backends::processStatusMessage(
        static_cast<fa_window::backends::ProcessStatus>(999))),
    std::logic_error);
}

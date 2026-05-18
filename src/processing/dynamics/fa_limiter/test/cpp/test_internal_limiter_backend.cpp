#include "fa_limiter/backends/internal_limiter.hpp"

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
  bytes.reserve(samples.size() * sizeof(float));
  for (const float sample : samples) {
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(float));
    std::memcpy(bytes.data() + offset, &sample, sizeof(float));
  }
  return bytes;
}

std::vector<float> decodeFloat32Le(const std::vector<uint8_t> & bytes)
{
  std::vector<float> samples;
  samples.reserve(bytes.size() / sizeof(float));
  for (size_t offset = 0; offset < bytes.size(); offset += sizeof(float)) {
    float sample = 0.0F;
    std::memcpy(&sample, bytes.data() + offset, sizeof(float));
    samples.push_back(sample);
  }
  return samples;
}

}  // namespace

TEST(InternalLimiterBackendContract, PassesSamplesWithinThreshold)
{
  fa_limiter::backends::InternalLimiterBackend backend(
    fa_limiter::backends::InternalLimiterConfig{1, 0.8});

  std::vector<uint8_t> output;
  const fa_limiter::backends::ProcessResult result =
    backend.process(float32LeBytes({0.0F, 0.5F, -0.5F}), output);

  EXPECT_EQ(result.status, fa_limiter::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_limited, 0U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.0F, 0.5F, -0.5F}));
}

TEST(InternalLimiterBackendContract, LimitsFiniteSamplesToExplicitThreshold)
{
  fa_limiter::backends::InternalLimiterBackend backend(
    fa_limiter::backends::InternalLimiterConfig{1, 0.5});

  std::vector<uint8_t> output;
  const fa_limiter::backends::ProcessResult result =
    backend.process(float32LeBytes({0.75F, -0.75F, 0.25F}), output);

  EXPECT_EQ(result.status, fa_limiter::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_limited, 2U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.5F, -0.5F, 0.25F}));
}

TEST(InternalLimiterBackendContract, LimitsAlreadyOutOfRangeFiniteSamplesByDesign)
{
  fa_limiter::backends::InternalLimiterBackend backend(
    fa_limiter::backends::InternalLimiterConfig{1, 1.0});

  std::vector<uint8_t> output;
  const fa_limiter::backends::ProcessResult result =
    backend.process(float32LeBytes({1.25F, -1.25F}), output);

  EXPECT_EQ(result.status, fa_limiter::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_limited, 2U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{1.0F, -1.0F}));
}

TEST(InternalLimiterBackendContract, ReportsInputRejectionStatuses)
{
  fa_limiter::backends::InternalLimiterBackend backend(
    fa_limiter::backends::InternalLimiterConfig{1, 1.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output).status,
    fa_limiter::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output).status,
    fa_limiter::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_limiter::backends::ProcessStatus::kNonFiniteInput);
}

TEST(InternalLimiterBackendContract, RejectsUnhandledStatusValues)
{
  EXPECT_THROW(
    fa_limiter::backends::processStatusMessage(
      static_cast<fa_limiter::backends::ProcessStatus>(999)),
    std::logic_error);
}

TEST(InternalLimiterBackendContract, RejectedFrameDoesNotOverwriteOutput)
{
  fa_limiter::backends::InternalLimiterBackend backend(
    fa_limiter::backends::InternalLimiterConfig{1, 1.0});

  std::vector<uint8_t> output = float32LeBytes({0.125F});
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::infinity()}), output).status,
    fa_limiter::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.125F}));
}

#include "fa_notch/backends/internal_notch.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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

TEST(InternalNotchBackendContract, FiltersFloat32LeWithBiquadState)
{
  fa_notch::backends::InternalNotchBackend backend(
    fa_notch::backends::InternalNotchConfig{1000, 1, 60.0, 30.0});

  std::vector<uint8_t> output;
  const auto status = backend.process(float32LeBytes({1.0F, 0.0F, 0.0F}), output);

  ASSERT_EQ(status, fa_notch::backends::ProcessStatus::kOk);
  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 3U);

  const fa_notch::backends::BiquadCoefficients & c = backend.coefficients();
  const double y0 = c.b0;
  const double y1 = c.b1 - (c.a1 * y0);
  const double y2 = c.b2 - (c.a1 * y1) - (c.a2 * y0);
  EXPECT_NEAR(samples.at(0), static_cast<float>(y0), 1.0e-6F);
  EXPECT_NEAR(samples.at(1), static_cast<float>(y1), 1.0e-6F);
  EXPECT_NEAR(samples.at(2), static_cast<float>(y2), 1.0e-6F);
}

TEST(InternalNotchBackendContract, ReportsInputRejectionStatuses)
{
  fa_notch::backends::InternalNotchBackend backend(
    fa_notch::backends::InternalNotchConfig{16000, 1, 60.0, 30.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output),
    fa_notch::backends::ProcessStatus::kEmptyInput);

  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output),
    fa_notch::backends::ProcessStatus::kMisalignedInput);

  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output),
    fa_notch::backends::ProcessStatus::kNonFiniteInput);
}

TEST(InternalNotchBackendContract, DoesNotCommitStateWhenFrameIsRejected)
{
  fa_notch::backends::InternalNotchBackend backend(
    fa_notch::backends::InternalNotchConfig{1000, 1, 60.0, 30.0});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process(float32LeBytes({1.0F, 0.0F}), output),
    fa_notch::backends::ProcessStatus::kOk);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output),
    fa_notch::backends::ProcessStatus::kNonFiniteInput);
  ASSERT_EQ(
    backend.process(float32LeBytes({0.0F}), output),
    fa_notch::backends::ProcessStatus::kOk);

  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 1U);

  const fa_notch::backends::BiquadCoefficients & c = backend.coefficients();
  const double y0 = c.b0;
  const double y1 = c.b1 - (c.a1 * y0);
  const double expected = c.b2 - (c.a1 * y1) - (c.a2 * y0);
  EXPECT_NEAR(samples.at(0), static_cast<float>(expected), 1.0e-6F);
}

TEST(InternalNotchBackendContract, DoesNotRequireNormalizedInputRange)
{
  fa_notch::backends::InternalNotchBackend backend(
    fa_notch::backends::InternalNotchConfig{16000, 1, 60.0, 30.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process(float32LeBytes({2.0F, -2.0F}), output),
    fa_notch::backends::ProcessStatus::kOk);
}

#include "fa_pan/backends/internal_constant_power_pan.hpp"

#include <cmath>
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
    fa_pan::backends::appendFloat32Le(sample, bytes);
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

TEST(InternalConstantPowerPanBackendContract, RejectsInvalidConfig)
{
  EXPECT_THROW(
    fa_pan::backends::InternalConstantPowerPanBackend(
      fa_pan::backends::InternalConstantPowerPanConfig{-1.25}),
    std::runtime_error);
  EXPECT_THROW(
    fa_pan::backends::InternalConstantPowerPanBackend(
      fa_pan::backends::InternalConstantPowerPanConfig{
        std::numeric_limits<double>::quiet_NaN()}),
    std::runtime_error);
}

TEST(InternalConstantPowerPanBackendContract, AppliesConstantPowerGains)
{
  fa_pan::backends::InternalConstantPowerPanBackend left_backend(
    fa_pan::backends::InternalConstantPowerPanConfig{-1.0});
  fa_pan::backends::InternalConstantPowerPanBackend center_backend(
    fa_pan::backends::InternalConstantPowerPanConfig{0.0});
  fa_pan::backends::InternalConstantPowerPanBackend right_backend(
    fa_pan::backends::InternalConstantPowerPanConfig{1.0});

  EXPECT_NEAR(left_backend.leftGain(), 1.0, 1.0e-12);
  EXPECT_NEAR(left_backend.rightGain(), 0.0, 1.0e-12);
  EXPECT_NEAR(center_backend.leftGain(), std::sqrt(0.5), 1.0e-12);
  EXPECT_NEAR(center_backend.rightGain(), std::sqrt(0.5), 1.0e-12);
  EXPECT_NEAR(right_backend.leftGain(), 0.0, 1.0e-12);
  EXPECT_NEAR(right_backend.rightGain(), 1.0, 1.0e-12);

  std::vector<uint8_t> output;
  ASSERT_EQ(
    center_backend.process(float32LeBytes({0.5F, 0.5F}), output),
    fa_pan::backends::ProcessStatus::kOk);
  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 2U);
  EXPECT_NEAR(samples.at(0), static_cast<float>(0.5 * std::sqrt(0.5)), 1.0e-6F);
  EXPECT_NEAR(samples.at(1), static_cast<float>(0.5 * std::sqrt(0.5)), 1.0e-6F);
}

TEST(InternalConstantPowerPanBackendContract, ReportsTypedInputAndOutputFailures)
{
  fa_pan::backends::InternalConstantPowerPanBackend backend(
    fa_pan::backends::InternalConstantPowerPanConfig{0.0});

  std::vector<uint8_t> output = float32LeBytes({0.25F, 0.25F});
  EXPECT_EQ(
    backend.process({}, output),
    fa_pan::backends::ProcessStatus::kInvalidInputSize);
  EXPECT_FALSE(output.empty());

  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output),
    fa_pan::backends::ProcessStatus::kInvalidInputSize);

  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN(), 0.0F}), output),
    fa_pan::backends::ProcessStatus::kInvalidInputSample);

  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F, 0.0F}), output),
    fa_pan::backends::ProcessStatus::kInvalidInputSample);
}

TEST(InternalConstantPowerPanBackendContract, BackendDoesNotMutateOutputOnFailure)
{
  fa_pan::backends::InternalConstantPowerPanBackend backend(
    fa_pan::backends::InternalConstantPowerPanConfig{0.0});

  const std::vector<uint8_t> previous = float32LeBytes({0.125F, -0.125F});
  std::vector<uint8_t> output = previous;
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::infinity(), 0.0F}), output),
    fa_pan::backends::ProcessStatus::kInvalidInputSample);

  EXPECT_EQ(output, previous);
}

TEST(InternalConstantPowerPanBackendContract, UnknownStatusIsInvariantError)
{
  EXPECT_THROW(
    {
      [[maybe_unused]] const char * message =
        fa_pan::backends::processStatusMessage(
          static_cast<fa_pan::backends::ProcessStatus>(99));
    },
    std::logic_error);
}

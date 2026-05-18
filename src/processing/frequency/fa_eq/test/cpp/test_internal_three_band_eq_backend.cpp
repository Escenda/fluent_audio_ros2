#include "fa_eq/backends/internal_three_band_eq.hpp"

#include <cmath>
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

TEST(InternalThreeBandEqBackendContract, FlatGainsPreserveCanonicalSamples)
{
  fa_eq::backends::InternalThreeBandEqBackend backend(
    fa_eq::backends::InternalThreeBandEqConfig{1000, 1, 100.0, 200.0, 0.0, 0.0, 0.0});

  std::vector<uint8_t> output;
  const std::vector<float> input{0.0F, 0.25F, -0.25F};
  ASSERT_EQ(
    backend.process(float32LeBytes(input), output, false),
    fa_eq::backends::ProcessStatus::kOk);
  EXPECT_EQ(decodeFloat32Le(output), input);
}

TEST(InternalThreeBandEqBackendContract, AppliesThreeBandGainMix)
{
  fa_eq::backends::InternalThreeBandEqBackend backend(
    fa_eq::backends::InternalThreeBandEqConfig{1000, 1, 100.0, 200.0, 3.0, 0.0, -3.0});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process(float32LeBytes({0.0F, 0.25F, 0.25F}), output, false),
    fa_eq::backends::ProcessStatus::kOk);

  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 3U);
  EXPECT_FLOAT_EQ(samples.at(0), 0.0F);

  const double low_gain = backend.gainLowLinear();
  const double high_gain = backend.gainHighLinear();
  const double low1 = backend.lowAlpha() * 0.25;
  const double high1 = backend.highAlpha() * 0.25;
  const double mid1 = 0.25 - low1 - high1;
  const double expected1 = (low1 * low_gain) + mid1 + (high1 * high_gain);
  EXPECT_NEAR(samples.at(1), static_cast<float>(expected1), 1.0e-6F);

  const double low2 = low1 + (backend.lowAlpha() * (0.25 - low1));
  const double high2 = backend.highAlpha() * high1;
  const double mid2 = 0.25 - low2 - high2;
  const double expected2 = (low2 * low_gain) + mid2 + (high2 * high_gain);
  EXPECT_NEAR(samples.at(2), static_cast<float>(expected2), 1.0e-6F);
}

TEST(InternalThreeBandEqBackendContract, ReportsInputRejectionStatuses)
{
  fa_eq::backends::InternalThreeBandEqBackend backend(
    fa_eq::backends::InternalThreeBandEqConfig{16000, 1, 250.0, 4000.0, 0.0, 0.0, 0.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output, false),
    fa_eq::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output, false),
    fa_eq::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output, false),
    fa_eq::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output, false),
    fa_eq::backends::ProcessStatus::kOutOfRangeInput);
}

TEST(InternalThreeBandEqBackendContract, RejectsUnhandledStatusValues)
{
  EXPECT_THROW(
    fa_eq::backends::processStatusMessage(
      static_cast<fa_eq::backends::ProcessStatus>(999)),
    std::logic_error);
}

TEST(InternalThreeBandEqBackendContract, DoesNotCommitResetStateWhenFrameIsRejected)
{
  fa_eq::backends::InternalThreeBandEqBackend backend(
    fa_eq::backends::InternalThreeBandEqConfig{1000, 1, 100.0, 200.0, 3.0, 0.0, -3.0});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process(float32LeBytes({0.0F, 0.25F}), output, false),
    fa_eq::backends::ProcessStatus::kOk);
  EXPECT_EQ(
    backend.process(float32LeBytes({2.0F}), output, true),
    fa_eq::backends::ProcessStatus::kOutOfRangeInput);
  ASSERT_EQ(
    backend.process(float32LeBytes({0.25F}), output, false),
    fa_eq::backends::ProcessStatus::kOk);

  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 1U);

  const double low_gain = backend.gainLowLinear();
  const double high_gain = backend.gainHighLinear();
  const double low1 = backend.lowAlpha() * 0.25;
  const double high1 = backend.highAlpha() * 0.25;
  const double low2 = low1 + (backend.lowAlpha() * (0.25 - low1));
  const double high2 = backend.highAlpha() * high1;
  const double mid2 = 0.25 - low2 - high2;
  const double expected = (low2 * low_gain) + mid2 + (high2 * high_gain);
  EXPECT_NEAR(samples.at(0), static_cast<float>(expected), 1.0e-6F);
}

TEST(InternalThreeBandEqBackendContract, ResetStateMakesNextFrameStartFresh)
{
  fa_eq::backends::InternalThreeBandEqBackend backend(
    fa_eq::backends::InternalThreeBandEqConfig{1000, 1, 100.0, 200.0, 3.0, 0.0, -3.0});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process(float32LeBytes({0.0F, 0.25F}), output, false),
    fa_eq::backends::ProcessStatus::kOk);
  ASSERT_EQ(
    backend.process(float32LeBytes({0.25F}), output, true),
    fa_eq::backends::ProcessStatus::kOk);

  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 1U);
  EXPECT_NEAR(samples.at(0), static_cast<float>(0.25 * backend.gainLowLinear()), 1.0e-6F);
}

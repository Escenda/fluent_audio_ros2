#include "fa_band_pass/backends/internal_first_order_band_pass.hpp"

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

TEST(InternalFirstOrderBandPassBackendContract, FiltersFloat32LeWithCascadedState)
{
  fa_band_pass::backends::InternalFirstOrderBandPassBackend backend(
    fa_band_pass::backends::InternalFirstOrderBandPassConfig{1000, 1, 100.0, 200.0});

  std::vector<uint8_t> output;
  const auto status = backend.process(float32LeBytes({0.0F, 1.0F, 1.0F}), output, false);

  ASSERT_EQ(status, fa_band_pass::backends::ProcessStatus::kOk);
  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 3U);
  EXPECT_FLOAT_EQ(samples.at(0), 0.0F);

  const double hp_alpha = backend.highPassAlpha();
  const double lp_alpha = backend.lowPassAlpha();
  const double second = hp_alpha * lp_alpha;
  const double third = second + (lp_alpha * ((hp_alpha * hp_alpha) - second));
  EXPECT_NEAR(samples.at(1), static_cast<float>(second), 1.0e-6F);
  EXPECT_NEAR(samples.at(2), static_cast<float>(third), 1.0e-6F);
}

TEST(InternalFirstOrderBandPassBackendContract, ReportsInputRejectionStatuses)
{
  fa_band_pass::backends::InternalFirstOrderBandPassBackend backend(
    fa_band_pass::backends::InternalFirstOrderBandPassConfig{16000, 1, 80.0, 3400.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output, false),
    fa_band_pass::backends::ProcessStatus::kEmptyInput);

  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output, false),
    fa_band_pass::backends::ProcessStatus::kMisalignedInput);

  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output, false),
    fa_band_pass::backends::ProcessStatus::kNonFiniteInput);

  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output, false),
    fa_band_pass::backends::ProcessStatus::kOutOfRangeInput);
}

TEST(InternalFirstOrderBandPassBackendContract, ResetStateIsCommittedOnlyOnSuccessfulFrame)
{
  fa_band_pass::backends::InternalFirstOrderBandPassBackend backend(
    fa_band_pass::backends::InternalFirstOrderBandPassConfig{1000, 1, 100.0, 200.0});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process(float32LeBytes({0.0F, 1.0F}), output, false),
    fa_band_pass::backends::ProcessStatus::kOk);

  EXPECT_EQ(
    backend.process(float32LeBytes({2.0F}), output, true),
    fa_band_pass::backends::ProcessStatus::kOutOfRangeInput);

  ASSERT_EQ(
    backend.process(float32LeBytes({1.0F}), output, false),
    fa_band_pass::backends::ProcessStatus::kOk);
  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 1U);

  const double hp_alpha = backend.highPassAlpha();
  const double lp_alpha = backend.lowPassAlpha();
  const double previous_lp = hp_alpha * lp_alpha;
  const double expected = previous_lp + (lp_alpha * ((hp_alpha * hp_alpha) - previous_lp));
  EXPECT_NEAR(samples.at(0), static_cast<float>(expected), 1.0e-6F);
}

TEST(InternalFirstOrderBandPassBackendContract, ResetStateMakesNextFrameStartFresh)
{
  fa_band_pass::backends::InternalFirstOrderBandPassBackend backend(
    fa_band_pass::backends::InternalFirstOrderBandPassConfig{1000, 1, 100.0, 200.0});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process(float32LeBytes({0.0F, 1.0F}), output, false),
    fa_band_pass::backends::ProcessStatus::kOk);
  ASSERT_EQ(
    backend.process(float32LeBytes({0.25F}), output, true),
    fa_band_pass::backends::ProcessStatus::kOk);

  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 1U);
  EXPECT_FLOAT_EQ(samples.at(0), 0.0F);
}

TEST(InternalFirstOrderBandPassBackendContract, RejectsOutOfRangeInputBeforeFiltering)
{
  fa_band_pass::backends::InternalFirstOrderBandPassBackend backend(
    fa_band_pass::backends::InternalFirstOrderBandPassConfig{48000, 1, 1.0, 2.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process(float32LeBytes({-1.25F, 1.25F}), output, false),
    fa_band_pass::backends::ProcessStatus::kOutOfRangeInput);
}

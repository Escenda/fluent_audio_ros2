#include "fa_deesser/backends/internal_split_band_deesser.hpp"

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

TEST(InternalSplitBandDeesserBackendContract, ThresholdAboveSignalPreservesSamples)
{
  fa_deesser::backends::InternalSplitBandDeesserBackend backend(
    fa_deesser::backends::InternalSplitBandDeesserConfig{1000, 1, 100.0, 1.0, -9.0});

  std::vector<uint8_t> output;
  const fa_deesser::backends::ProcessResult result =
    backend.process(float32LeBytes({0.0F, 0.25F, -0.25F}), output, false);

  EXPECT_EQ(result.status, fa_deesser::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_attenuated, 0U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.0F, 0.25F, -0.25F}));
}

TEST(InternalSplitBandDeesserBackendContract, AttenuatesHighBandAboveThreshold)
{
  fa_deesser::backends::InternalSplitBandDeesserBackend backend(
    fa_deesser::backends::InternalSplitBandDeesserConfig{1000, 1, 100.0, 0.05, -6.0});

  std::vector<uint8_t> output;
  const fa_deesser::backends::ProcessResult result =
    backend.process(float32LeBytes({0.0F, 0.25F, 0.25F}), output, false);

  ASSERT_EQ(result.status, fa_deesser::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_attenuated, 2U);

  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 3U);
  EXPECT_FLOAT_EQ(samples.at(0), 0.0F);

  const double low1 = backend.alpha() * 0.25;
  const double high1 = 0.25 - low1;
  const double expected1 = low1 + (high1 * backend.attenuationGain());
  EXPECT_NEAR(samples.at(1), static_cast<float>(expected1), 1.0e-6F);

  const double low2 = low1 + (backend.alpha() * (0.25 - low1));
  const double high2 = 0.25 - low2;
  const double expected2 = low2 + (high2 * backend.attenuationGain());
  EXPECT_NEAR(samples.at(2), static_cast<float>(expected2), 1.0e-6F);
}

TEST(InternalSplitBandDeesserBackendContract, ReportsInputRejectionStatuses)
{
  fa_deesser::backends::InternalSplitBandDeesserBackend backend(
    fa_deesser::backends::InternalSplitBandDeesserConfig{16000, 1, 4500.0, 0.08, -9.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output, false).status,
    fa_deesser::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output, false).status,
    fa_deesser::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(
      float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output, false).status,
    fa_deesser::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output, false).status,
    fa_deesser::backends::ProcessStatus::kOutOfRangeInput);
}

TEST(InternalSplitBandDeesserBackendContract, DoesNotCommitResetStateWhenFrameIsRejected)
{
  fa_deesser::backends::InternalSplitBandDeesserBackend backend(
    fa_deesser::backends::InternalSplitBandDeesserConfig{1000, 1, 100.0, 0.05, -6.0});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process(float32LeBytes({0.0F, 0.25F}), output, false).status,
    fa_deesser::backends::ProcessStatus::kOk);
  EXPECT_EQ(
    backend.process(float32LeBytes({2.0F}), output, true).status,
    fa_deesser::backends::ProcessStatus::kOutOfRangeInput);
  ASSERT_EQ(
    backend.process(float32LeBytes({0.25F}), output, false).status,
    fa_deesser::backends::ProcessStatus::kOk);

  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 1U);
  const double low1 = backend.alpha() * 0.25;
  const double low2 = low1 + (backend.alpha() * (0.25 - low1));
  const double high2 = 0.25 - low2;
  const double expected = low2 + (high2 * backend.attenuationGain());
  EXPECT_NEAR(samples.at(0), static_cast<float>(expected), 1.0e-6F);
}

TEST(InternalSplitBandDeesserBackendContract, ResetStateMakesNextFrameStartFresh)
{
  fa_deesser::backends::InternalSplitBandDeesserBackend backend(
    fa_deesser::backends::InternalSplitBandDeesserConfig{1000, 1, 100.0, 0.05, -6.0});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process(float32LeBytes({0.0F, 0.25F}), output, false).status,
    fa_deesser::backends::ProcessStatus::kOk);
  ASSERT_EQ(
    backend.process(float32LeBytes({0.25F}), output, true).status,
    fa_deesser::backends::ProcessStatus::kOk);

  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 1U);
  const double low = backend.alpha() * 0.25;
  const double high = 0.25 - low;
  const double expected = low + (high * backend.attenuationGain());
  EXPECT_NEAR(samples.at(0), static_cast<float>(expected), 1.0e-6F);
}

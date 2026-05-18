#include "fa_sidechain/backends/internal_sidechain_detector.hpp"

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
  std::vector<uint8_t> bytes(samples.size() * sizeof(float), 0U);
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

float readFloat32Le(const std::vector<uint8_t> & bytes)
{
  const uint32_t bits =
    static_cast<uint32_t>(bytes[0]) |
    (static_cast<uint32_t>(bytes[1]) << 8U) |
    (static_cast<uint32_t>(bytes[2]) << 16U) |
    (static_cast<uint32_t>(bytes[3]) << 24U);
  float sample = 0.0F;
  std::memcpy(&sample, &bits, sizeof(sample));
  return sample;
}

fa_sidechain::backends::InternalSidechainDetectorBackend makeBackend()
{
  return fa_sidechain::backends::InternalSidechainDetectorBackend(
    fa_sidechain::backends::InternalSidechainDetectorConfig{1, 0.05, -12.0, 0.0});
}

}  // namespace

TEST(InternalSidechainDetectorBackendContract, PublishesActiveGainForThresholdRms)
{
  auto backend = makeBackend();
  std::vector<uint8_t> control_data;

  const auto result = backend.detect(float32LeBytes({0.05F, -0.05F}), control_data);

  EXPECT_EQ(result.status, fa_sidechain::backends::ProcessStatus::kOk);
  EXPECT_TRUE(result.active);
  EXPECT_NEAR(result.rms, 0.05, 1.0e-6);
  EXPECT_NEAR(result.gain_linear, fa_sidechain::backends::dbToLinear(-12.0), 1.0e-6);
  ASSERT_EQ(control_data.size(), sizeof(float));
  EXPECT_NEAR(readFloat32Le(control_data), static_cast<float>(result.gain_linear), 1.0e-6F);
  EXPECT_NEAR(backend.lastRms(), result.rms, 1.0e-6);
  EXPECT_NEAR(backend.lastGainLinear(), result.gain_linear, 1.0e-6);
  EXPECT_TRUE(backend.lastActive());
}

TEST(InternalSidechainDetectorBackendContract, PublishesInactiveGainBelowThreshold)
{
  auto backend = makeBackend();
  std::vector<uint8_t> control_data;

  const auto result = backend.detect(float32LeBytes({0.03F, -0.04F}), control_data);

  EXPECT_EQ(result.status, fa_sidechain::backends::ProcessStatus::kOk);
  EXPECT_FALSE(result.active);
  EXPECT_NEAR(result.gain_linear, 1.0, 1.0e-6);
  ASSERT_EQ(control_data.size(), sizeof(float));
  EXPECT_NEAR(readFloat32Le(control_data), 1.0F, 1.0e-6F);
}

TEST(InternalSidechainDetectorBackendContract, RejectsInvalidInputWithoutStateCommit)
{
  auto backend = makeBackend();
  std::vector<uint8_t> control_data;
  ASSERT_EQ(
    backend.detect(float32LeBytes({0.1F}), control_data).status,
    fa_sidechain::backends::ProcessStatus::kOk);
  const std::vector<uint8_t> previous_control_data = control_data;

  EXPECT_EQ(
    backend.detect(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), control_data).status,
    fa_sidechain::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(control_data, previous_control_data);
  EXPECT_NEAR(backend.lastRms(), 0.1, 1.0e-6);
  EXPECT_TRUE(backend.lastActive());
}

TEST(InternalSidechainDetectorBackendContract, RejectsInvalidConfiguration)
{
  EXPECT_THROW(
    fa_sidechain::backends::InternalSidechainDetectorBackend(
      fa_sidechain::backends::InternalSidechainDetectorConfig{0, 0.05, -12.0, 0.0}),
    std::runtime_error);
  EXPECT_THROW(
    fa_sidechain::backends::InternalSidechainDetectorBackend(
      fa_sidechain::backends::InternalSidechainDetectorConfig{1, 0.0, -12.0, 0.0}),
    std::runtime_error);
  EXPECT_THROW(
    fa_sidechain::backends::InternalSidechainDetectorBackend(
      fa_sidechain::backends::InternalSidechainDetectorConfig{1, 0.05, 100.0, 0.0}),
    std::runtime_error);
}

TEST(InternalSidechainDetectorBackendContract, FailsClosedForUnknownStatus)
{
  EXPECT_THROW(
    static_cast<void>(
      fa_sidechain::backends::processStatusMessage(
        static_cast<fa_sidechain::backends::ProcessStatus>(999))),
    std::logic_error);
}

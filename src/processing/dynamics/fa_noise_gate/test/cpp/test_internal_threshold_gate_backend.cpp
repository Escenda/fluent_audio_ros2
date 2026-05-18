#include "fa_noise_gate/backends/internal_threshold_gate.hpp"

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

TEST(InternalThresholdGateBackendContract, PassesSamplesAtOrAboveThreshold)
{
  fa_noise_gate::backends::InternalThresholdGateBackend backend(
    fa_noise_gate::backends::InternalThresholdGateConfig{1, 0.25, 0.0});

  std::vector<uint8_t> output;
  const fa_noise_gate::backends::ProcessResult result =
    backend.process(float32LeBytes({0.25F, -0.25F, 0.5F}), output);

  EXPECT_EQ(result.status, fa_noise_gate::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_gated, 0U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.25F, -0.25F, 0.5F}));
}

TEST(InternalThresholdGateBackendContract, AppliesClosedGainBelowThreshold)
{
  fa_noise_gate::backends::InternalThresholdGateBackend backend(
    fa_noise_gate::backends::InternalThresholdGateConfig{1, 0.25, 0.5});

  std::vector<uint8_t> output;
  const fa_noise_gate::backends::ProcessResult result =
    backend.process(float32LeBytes({0.10F, -0.10F, 0.30F}), output);

  EXPECT_EQ(result.status, fa_noise_gate::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_gated, 2U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.05F, -0.05F, 0.30F}));
}

TEST(InternalThresholdGateBackendContract, AllowsZeroClosedGainAsExplicitMute)
{
  fa_noise_gate::backends::InternalThresholdGateBackend backend(
    fa_noise_gate::backends::InternalThresholdGateConfig{1, 0.25, 0.0});

  std::vector<uint8_t> output;
  const fa_noise_gate::backends::ProcessResult result =
    backend.process(float32LeBytes({0.10F, -0.10F}), output);

  EXPECT_EQ(result.status, fa_noise_gate::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_gated, 2U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.0F, -0.0F}));
}

TEST(InternalThresholdGateBackendContract, ReportsInputRejectionStatuses)
{
  fa_noise_gate::backends::InternalThresholdGateBackend backend(
    fa_noise_gate::backends::InternalThresholdGateConfig{1, 0.25, 0.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output).status,
    fa_noise_gate::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output).status,
    fa_noise_gate::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_noise_gate::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output).status,
    fa_noise_gate::backends::ProcessStatus::kOutOfRangeInput);
}

TEST(InternalThresholdGateBackendContract, RejectsUnhandledStatusValues)
{
  EXPECT_THROW(
    fa_noise_gate::backends::processStatusMessage(
      static_cast<fa_noise_gate::backends::ProcessStatus>(999)),
    std::logic_error);
}

TEST(InternalThresholdGateBackendContract, RejectedFrameDoesNotOverwriteOutput)
{
  fa_noise_gate::backends::InternalThresholdGateBackend backend(
    fa_noise_gate::backends::InternalThresholdGateConfig{1, 0.25, 0.0});

  std::vector<uint8_t> output = float32LeBytes({0.125F});
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::infinity()}), output).status,
    fa_noise_gate::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.125F}));
}

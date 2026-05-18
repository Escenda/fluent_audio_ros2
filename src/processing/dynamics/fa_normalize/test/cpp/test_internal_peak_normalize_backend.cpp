#include "fa_normalize/backends/internal_peak_normalize.hpp"

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

TEST(InternalPeakNormalizeBackendContract, RejectsInvalidConfig)
{
  using fa_normalize::backends::InternalPeakNormalizeBackend;
  using fa_normalize::backends::InternalPeakNormalizeConfig;

  EXPECT_THROW(
    InternalPeakNormalizeBackend(InternalPeakNormalizeConfig{0, 0.9, 0.0001}),
    std::runtime_error);
  EXPECT_THROW(
    InternalPeakNormalizeBackend(InternalPeakNormalizeConfig{1, 0.0, 0.0001}),
    std::runtime_error);
  EXPECT_THROW(
    InternalPeakNormalizeBackend(InternalPeakNormalizeConfig{1, 1.1, 0.0001}),
    std::runtime_error);
  EXPECT_THROW(
    InternalPeakNormalizeBackend(InternalPeakNormalizeConfig{1, 0.9, -0.1}),
    std::runtime_error);
  EXPECT_THROW(
    InternalPeakNormalizeBackend(InternalPeakNormalizeConfig{1, 0.9, 0.9}),
    std::runtime_error);
}

TEST(InternalPeakNormalizeBackendContract, NormalizesFramePeakToTarget)
{
  fa_normalize::backends::InternalPeakNormalizeBackend backend(
    fa_normalize::backends::InternalPeakNormalizeConfig{1, 0.8, 0.0001});

  std::vector<uint8_t> output;
  const fa_normalize::backends::ProcessResult result =
    backend.process(float32LeBytes({0.2F, -0.4F, 0.1F}), output);

  EXPECT_EQ(result.status, fa_normalize::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.mode, fa_normalize::backends::ProcessMode::kNormalized);
  EXPECT_NEAR(result.peak, 0.4, 1.0e-6);
  EXPECT_NEAR(result.gain, 2.0, 1.0e-6);
  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 3U);
  EXPECT_NEAR(samples[0], 0.4F, 1.0e-6F);
  EXPECT_NEAR(samples[1], -0.8F, 1.0e-6F);
  EXPECT_NEAR(samples[2], 0.2F, 1.0e-6F);
}

TEST(InternalPeakNormalizeBackendContract, SilencePassesThroughWithUnityGain)
{
  fa_normalize::backends::InternalPeakNormalizeBackend backend(
    fa_normalize::backends::InternalPeakNormalizeConfig{1, 0.8, 0.1});

  std::vector<uint8_t> output;
  const std::vector<uint8_t> input = float32LeBytes({0.05F, -0.02F, 0.0F});
  const fa_normalize::backends::ProcessResult result = backend.process(input, output);

  EXPECT_EQ(result.status, fa_normalize::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.mode, fa_normalize::backends::ProcessMode::kSilencePassthrough);
  EXPECT_NEAR(result.peak, 0.05, 1.0e-6);
  EXPECT_DOUBLE_EQ(result.gain, 1.0);
  EXPECT_EQ(output, input);
}

TEST(InternalPeakNormalizeBackendContract, ReportsInputRejectionStatuses)
{
  fa_normalize::backends::InternalPeakNormalizeBackend backend(
    fa_normalize::backends::InternalPeakNormalizeConfig{1, 0.8, 0.0001});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output).status,
    fa_normalize::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output).status,
    fa_normalize::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_normalize::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output).status,
    fa_normalize::backends::ProcessStatus::kOutOfRangeInput);
}

TEST(InternalPeakNormalizeBackendContract, RejectedFrameDoesNotOverwriteOutput)
{
  fa_normalize::backends::InternalPeakNormalizeBackend backend(
    fa_normalize::backends::InternalPeakNormalizeConfig{1, 0.8, 0.0001});

  std::vector<uint8_t> output = float32LeBytes({0.125F});
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output).status,
    fa_normalize::backends::ProcessStatus::kOutOfRangeInput);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.125F}));
}

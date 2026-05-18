#include "fa_declick/backends/internal_impulse_declick.hpp"

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

TEST(InternalImpulseDeclickBackendContract, RejectsInvalidConfig)
{
  using fa_declick::backends::InternalImpulseDeclickBackend;
  using fa_declick::backends::InternalImpulseDeclickConfig;

  EXPECT_THROW(InternalImpulseDeclickBackend(InternalImpulseDeclickConfig{0, 0.25, 1}), std::runtime_error);
  EXPECT_THROW(InternalImpulseDeclickBackend(InternalImpulseDeclickConfig{1, 0.0, 1}), std::runtime_error);
  EXPECT_THROW(InternalImpulseDeclickBackend(InternalImpulseDeclickConfig{1, 2.1, 1}), std::runtime_error);
  EXPECT_THROW(
    InternalImpulseDeclickBackend(
      InternalImpulseDeclickConfig{1, std::numeric_limits<double>::quiet_NaN(), 1}),
    std::runtime_error);
  EXPECT_THROW(InternalImpulseDeclickBackend(InternalImpulseDeclickConfig{1, 0.25, 0}), std::runtime_error);
}

TEST(InternalImpulseDeclickBackendContract, CorrectsSingleSampleMonoClick)
{
  fa_declick::backends::InternalImpulseDeclickBackend backend(
    fa_declick::backends::InternalImpulseDeclickConfig{1, 0.25, 1});

  std::vector<uint8_t> output;
  const fa_declick::backends::ProcessResult result =
    backend.process(float32LeBytes({0.1F, 0.9F, 0.1F}), output);

  EXPECT_EQ(result.status, fa_declick::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_corrected, 1U);
  EXPECT_EQ(result.click_runs_corrected, 1U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.1F, 0.1F, 0.1F}));
}

TEST(InternalImpulseDeclickBackendContract, CorrectsBoundedMultiSampleRun)
{
  fa_declick::backends::InternalImpulseDeclickBackend backend(
    fa_declick::backends::InternalImpulseDeclickConfig{1, 0.25, 2});

  std::vector<uint8_t> output;
  const fa_declick::backends::ProcessResult result =
    backend.process(float32LeBytes({0.0F, 0.8F, -0.8F, 0.0F}), output);

  EXPECT_EQ(result.status, fa_declick::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_corrected, 2U);
  EXPECT_EQ(result.click_runs_corrected, 1U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F}));
}

TEST(InternalImpulseDeclickBackendContract, HandlesStereoChannelsIndependently)
{
  fa_declick::backends::InternalImpulseDeclickBackend backend(
    fa_declick::backends::InternalImpulseDeclickConfig{2, 0.25, 1});

  std::vector<uint8_t> output;
  const fa_declick::backends::ProcessResult result =
    backend.process(float32LeBytes({0.0F, 0.2F, 0.9F, 0.2F, 0.0F, 0.2F}), output);

  EXPECT_EQ(result.status, fa_declick::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_corrected, 1U);
  EXPECT_EQ(result.click_runs_corrected, 1U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.0F, 0.2F, 0.0F, 0.2F, 0.0F, 0.2F}));
}

TEST(InternalImpulseDeclickBackendContract, ReportsInputRejectionStatuses)
{
  fa_declick::backends::InternalImpulseDeclickBackend backend(
    fa_declick::backends::InternalImpulseDeclickConfig{1, 0.25, 1});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output).status,
    fa_declick::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output).status,
    fa_declick::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_declick::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output).status,
    fa_declick::backends::ProcessStatus::kOutOfRangeInput);
}

TEST(InternalImpulseDeclickBackendContract, RejectedFrameDoesNotOverwriteOutput)
{
  fa_declick::backends::InternalImpulseDeclickBackend backend(
    fa_declick::backends::InternalImpulseDeclickConfig{1, 0.25, 1});

  std::vector<uint8_t> output = float32LeBytes({0.125F});
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_declick::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.125F}));
}

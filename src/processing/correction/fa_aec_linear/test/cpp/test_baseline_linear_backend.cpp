#include "fa_aec_linear/backends/baseline_linear.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace
{

std::vector<uint8_t> pcm16LeBytes(const std::vector<int16_t> & samples)
{
  std::vector<uint8_t> bytes;
  bytes.reserve(samples.size() * sizeof(int16_t));
  for (const int16_t sample : samples) {
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(int16_t));
    std::memcpy(bytes.data() + offset, &sample, sizeof(int16_t));
  }
  return bytes;
}

std::vector<int16_t> decodePcm16Le(const std::vector<uint8_t> & bytes)
{
  std::vector<int16_t> samples;
  samples.reserve(bytes.size() / sizeof(int16_t));
  for (size_t offset = 0; offset < bytes.size(); offset += sizeof(int16_t)) {
    int16_t sample = 0;
    std::memcpy(&sample, bytes.data() + offset, sizeof(int16_t));
    samples.push_back(sample);
  }
  return samples;
}

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

TEST(BaselineLinearBackendContract, RejectsInvalidConfig)
{
  using fa_aec_linear::backends::BaselineLinearBackend;
  using fa_aec_linear::backends::BaselineLinearConfig;

  EXPECT_THROW(BaselineLinearBackend(BaselineLinearConfig{0, "PCM16LE", 16, 1.0}), std::runtime_error);
  EXPECT_THROW(BaselineLinearBackend(BaselineLinearConfig{1, "PCM24LE", 24, 1.0}), std::runtime_error);
  EXPECT_THROW(
    BaselineLinearBackend(
      BaselineLinearConfig{1, "PCM16LE", 16, std::numeric_limits<double>::quiet_NaN()}),
    std::runtime_error);
}

TEST(BaselineLinearBackendContract, SubtractsPcm16Reference)
{
  fa_aec_linear::backends::BaselineLinearBackend backend(
    fa_aec_linear::backends::BaselineLinearConfig{1, "PCM16LE", 16, 1.0});

  std::vector<uint8_t> output;
  const fa_aec_linear::backends::ProcessResult result =
    backend.process(pcm16LeBytes({16384, 8192}), pcm16LeBytes({8192, 4096}), output);

  EXPECT_EQ(result.status, fa_aec_linear::backends::ProcessStatus::kOk);
  EXPECT_EQ(decodePcm16Le(output), (std::vector<int16_t>{8192, 4096}));
}

TEST(BaselineLinearBackendContract, SubtractsFloat32Reference)
{
  fa_aec_linear::backends::BaselineLinearBackend backend(
    fa_aec_linear::backends::BaselineLinearConfig{1, "FLOAT32LE", 32, 1.0});

  std::vector<uint8_t> output;
  const fa_aec_linear::backends::ProcessResult result =
    backend.process(float32LeBytes({0.5F, 0.0F}), float32LeBytes({0.25F, -0.5F}), output);

  EXPECT_EQ(result.status, fa_aec_linear::backends::ProcessStatus::kOk);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.25F, 0.5F}));
}

TEST(BaselineLinearBackendContract, ReportsInputRejectionStatuses)
{
  fa_aec_linear::backends::BaselineLinearBackend backend(
    fa_aec_linear::backends::BaselineLinearConfig{1, "FLOAT32LE", 32, 1.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, float32LeBytes({0.0F}), output).status,
    fa_aec_linear::backends::ProcessStatus::kEmptyMic);
  EXPECT_EQ(
    backend.process(float32LeBytes({0.0F}), {}, output).status,
    fa_aec_linear::backends::ProcessStatus::kEmptyReference);
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, float32LeBytes({0.0F}), output).status,
    fa_aec_linear::backends::ProcessStatus::kMisalignedMic);
  EXPECT_EQ(
    backend.process(float32LeBytes({0.0F}), std::vector<uint8_t>{0, 1, 2}, output).status,
    fa_aec_linear::backends::ProcessStatus::kMisalignedReference);
  EXPECT_EQ(
    backend.process(
      float32LeBytes({std::numeric_limits<float>::quiet_NaN()}),
      float32LeBytes({0.0F}),
      output).status,
    fa_aec_linear::backends::ProcessStatus::kNonFiniteMic);
  EXPECT_EQ(
    backend.process(float32LeBytes({0.0F}), float32LeBytes({1.25F}), output).status,
    fa_aec_linear::backends::ProcessStatus::kOutOfRangeReference);
}

TEST(BaselineLinearBackendContract, ReportsSampleCountMismatch)
{
  fa_aec_linear::backends::BaselineLinearBackend backend(
    fa_aec_linear::backends::BaselineLinearConfig{1, "FLOAT32LE", 32, 1.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process(float32LeBytes({0.0F, 0.25F}), float32LeBytes({0.0F}), output).status,
    fa_aec_linear::backends::ProcessStatus::kSampleCountMismatch);
}

TEST(BaselineLinearBackendContract, RejectsOutOfRangeOutputWithoutOverwritingOutput)
{
  fa_aec_linear::backends::BaselineLinearBackend backend(
    fa_aec_linear::backends::BaselineLinearConfig{1, "FLOAT32LE", 32, 1.0});

  std::vector<uint8_t> output = float32LeBytes({0.125F});
  EXPECT_EQ(
    backend.process(float32LeBytes({1.0F}), float32LeBytes({-1.0F}), output).status,
    fa_aec_linear::backends::ProcessStatus::kOutOfRangeOutput);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.125F}));
}

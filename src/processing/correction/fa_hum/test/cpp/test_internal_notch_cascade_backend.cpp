#include "fa_hum/backends/internal_notch_cascade.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace
{

fa_hum::backends::InternalNotchCascadeConfig validConfig()
{
  return fa_hum::backends::InternalNotchCascadeConfig{16000, 1, 60.0, 4, 30.0};
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

TEST(InternalNotchCascadeBackendContract, RejectsInvalidConfig)
{
  EXPECT_THROW(
    fa_hum::backends::InternalNotchCascadeBackend(
      fa_hum::backends::InternalNotchCascadeConfig{0, 1, 60.0, 4, 30.0}),
    std::runtime_error);
  EXPECT_THROW(
    fa_hum::backends::InternalNotchCascadeBackend(
      fa_hum::backends::InternalNotchCascadeConfig{16000, 0, 60.0, 4, 30.0}),
    std::runtime_error);
  EXPECT_THROW(
    fa_hum::backends::InternalNotchCascadeBackend(
      fa_hum::backends::InternalNotchCascadeConfig{16000, 1, 8000.0, 4, 30.0}),
    std::runtime_error);
  EXPECT_THROW(
    fa_hum::backends::InternalNotchCascadeBackend(
      fa_hum::backends::InternalNotchCascadeConfig{16000, 1, 60.0, 0, 30.0}),
    std::runtime_error);
  EXPECT_THROW(
    fa_hum::backends::InternalNotchCascadeBackend(
      fa_hum::backends::InternalNotchCascadeConfig{16000, 1, 60.0, 4, 0.0}),
    std::runtime_error);
}

TEST(InternalNotchCascadeBackendContract, BuildsHarmonicStagesBelowNyquist)
{
  fa_hum::backends::InternalNotchCascadeBackend backend(validConfig());

  EXPECT_EQ(backend.stageCount(), 4U);
  EXPECT_EQ(backend.centerFrequenciesHz(), (std::vector<double>{60.0, 120.0, 180.0, 240.0}));
}

TEST(InternalNotchCascadeBackendContract, ProcessesNormalizedFloat32Le)
{
  fa_hum::backends::InternalNotchCascadeBackend backend(validConfig());

  std::vector<uint8_t> output;
  const fa_hum::backends::ProcessResult result =
    backend.process("mic-a", 7U, float32LeBytes({0.1F, -0.1F, 0.2F, -0.2F}), output);

  EXPECT_EQ(result.status, fa_hum::backends::ProcessStatus::kOk);
  EXPECT_FALSE(result.source_reset);
  EXPECT_FALSE(result.epoch_reset);
  EXPECT_EQ(backend.activeSourceId(), "mic-a");
  EXPECT_TRUE(backend.hasActiveStream());
  EXPECT_EQ(backend.activeEpoch(), 7U);
  ASSERT_EQ(output.size(), 4U * sizeof(float));
  for (const float sample : decodeFloat32Le(output)) {
    EXPECT_TRUE(std::isfinite(sample));
    EXPECT_GE(sample, -1.0F);
    EXPECT_LE(sample, 1.0F);
  }
}

TEST(InternalNotchCascadeBackendContract, ResetsOnSourceAndEpochChanges)
{
  fa_hum::backends::InternalNotchCascadeBackend backend(validConfig());
  std::vector<uint8_t> output;

  EXPECT_EQ(
    backend.process("mic-a", 1U, float32LeBytes({0.1F, 0.2F}), output).status,
    fa_hum::backends::ProcessStatus::kOk);

  const fa_hum::backends::ProcessResult epoch_result =
    backend.process("mic-a", 2U, float32LeBytes({0.1F, 0.2F}), output);
  EXPECT_EQ(epoch_result.status, fa_hum::backends::ProcessStatus::kOk);
  EXPECT_FALSE(epoch_result.source_reset);
  EXPECT_TRUE(epoch_result.epoch_reset);
  EXPECT_EQ(backend.activeEpoch(), 2U);

  const fa_hum::backends::ProcessResult source_result =
    backend.process("mic-b", 1U, float32LeBytes({0.1F, 0.2F}), output);
  EXPECT_EQ(source_result.status, fa_hum::backends::ProcessStatus::kOk);
  EXPECT_TRUE(source_result.source_reset);
  EXPECT_FALSE(source_result.epoch_reset);
  EXPECT_EQ(backend.activeSourceId(), "mic-b");
}

TEST(InternalNotchCascadeBackendContract, RejectsStaleEpochAndBadInput)
{
  fa_hum::backends::InternalNotchCascadeBackend backend(validConfig());
  std::vector<uint8_t> output = float32LeBytes({0.25F});

  EXPECT_EQ(
    backend.process("mic-a", 2U, float32LeBytes({0.1F, 0.2F}), output).status,
    fa_hum::backends::ProcessStatus::kOk);
  output = float32LeBytes({0.25F});
  EXPECT_EQ(
    backend.process("mic-a", 1U, float32LeBytes({0.1F, 0.2F}), output).status,
    fa_hum::backends::ProcessStatus::kStaleEpoch);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.25F}));

  EXPECT_EQ(
    backend.process("", 2U, float32LeBytes({0.1F}), output).status,
    fa_hum::backends::ProcessStatus::kEmptySourceId);
  EXPECT_EQ(
    backend.process("mic-a", 2U, {}, output).status,
    fa_hum::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process("mic-a", 2U, std::vector<uint8_t>{0, 1, 2}, output).status,
    fa_hum::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(
      "mic-a",
      2U,
      float32LeBytes({std::numeric_limits<float>::quiet_NaN()}),
      output).status,
    fa_hum::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process("mic-a", 2U, float32LeBytes({1.25F}), output).status,
    fa_hum::backends::ProcessStatus::kOutOfRangeInput);
}

TEST(InternalNotchCascadeBackendContract, BackendDoesNotExposeRosTypes)
{
  EXPECT_STREQ(
    fa_hum::backends::processStatusMessage(
      fa_hum::backends::ProcessStatus::kStaleEpoch),
    "input epoch is older than active stream epoch");
  EXPECT_THROW(
    (void)fa_hum::backends::processStatusMessage(
      static_cast<fa_hum::backends::ProcessStatus>(999)),
    std::logic_error);
}

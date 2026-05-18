#include "fa_agc/backends/internal_rms_agc.hpp"

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

TEST(InternalRmsAgcBackendContract, RejectsInvalidConfig)
{
  using fa_agc::backends::InternalRmsAgcBackend;
  using fa_agc::backends::InternalRmsAgcConfig;

  EXPECT_THROW(
    InternalRmsAgcBackend(InternalRmsAgcConfig{-1, 1, 0.1, 0.25, 4.0, 10.0, 250.0}),
    std::runtime_error);
  EXPECT_THROW(
    InternalRmsAgcBackend(InternalRmsAgcConfig{16000, 0, 0.1, 0.25, 4.0, 10.0, 250.0}),
    std::runtime_error);
  EXPECT_THROW(
    InternalRmsAgcBackend(InternalRmsAgcConfig{16000, 1, 0.0, 0.25, 4.0, 10.0, 250.0}),
    std::runtime_error);
  EXPECT_THROW(
    InternalRmsAgcBackend(InternalRmsAgcConfig{16000, 1, 0.1, 1.25, 4.0, 10.0, 250.0}),
    std::runtime_error);
  EXPECT_THROW(
    InternalRmsAgcBackend(InternalRmsAgcConfig{16000, 1, 0.1, 0.25, 0.5, 10.0, 250.0}),
    std::runtime_error);
  EXPECT_THROW(
    InternalRmsAgcBackend(InternalRmsAgcConfig{16000, 1, 0.1, 0.25, 4.0, 0.0, 250.0}),
    std::runtime_error);
  EXPECT_THROW(
    InternalRmsAgcBackend(InternalRmsAgcConfig{16000, 1, 0.1, 0.25, 4.0, 10.0, 0.0}),
    std::runtime_error);
}

TEST(InternalRmsAgcBackendContract, PassesStableUnityGainWhenTargetMatchesRms)
{
  fa_agc::backends::InternalRmsAgcBackend backend(
    fa_agc::backends::InternalRmsAgcConfig{10, 1, 0.5, 0.25, 4.0, 10.0, 10.0});

  std::vector<uint8_t> output;
  const fa_agc::backends::ProcessResult result =
    backend.process(float32LeBytes({0.5F, -0.5F, 0.5F, -0.5F}), output);

  EXPECT_EQ(result.status, fa_agc::backends::ProcessStatus::kOk);
  EXPECT_DOUBLE_EQ(result.frame_rms, 0.5);
  EXPECT_DOUBLE_EQ(result.target_gain, 1.0);
  EXPECT_DOUBLE_EQ(result.committed_gain, 1.0);
  EXPECT_EQ(result.gain_direction, fa_agc::backends::GainDirection::kUnchanged);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.5F, -0.5F, 0.5F, -0.5F}));
}

TEST(InternalRmsAgcBackendContract, SilenceTargetsMaxGainAndCommitsIncrease)
{
  fa_agc::backends::InternalRmsAgcBackend backend(
    fa_agc::backends::InternalRmsAgcConfig{10, 1, 0.1, 0.25, 4.0, 10.0, 1.0});

  std::vector<uint8_t> output;
  const fa_agc::backends::ProcessResult result =
    backend.process(float32LeBytes({0.0F, 0.0F, 0.0F, 0.0F}), output);

  EXPECT_EQ(result.status, fa_agc::backends::ProcessStatus::kOk);
  EXPECT_DOUBLE_EQ(result.frame_rms, 0.0);
  EXPECT_DOUBLE_EQ(result.target_gain, 4.0);
  EXPECT_GT(result.committed_gain, 1.0);
  EXPECT_EQ(result.gain_direction, fa_agc::backends::GainDirection::kIncrease);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F}));
}

TEST(InternalRmsAgcBackendContract, ReportsInputRejectionStatuses)
{
  fa_agc::backends::InternalRmsAgcBackend backend(
    fa_agc::backends::InternalRmsAgcConfig{10, 1, 0.5, 0.25, 4.0, 10.0, 10.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output).status,
    fa_agc::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output).status,
    fa_agc::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_agc::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output).status,
    fa_agc::backends::ProcessStatus::kOutOfRangeInput);
}

TEST(InternalRmsAgcBackendContract, AttackAndReleaseUseConfiguredTimeConstants)
{
  fa_agc::backends::InternalRmsAgcBackend backend(
    fa_agc::backends::InternalRmsAgcConfig{1000, 1, 0.5, 0.25, 4.0, 10.0, 1000.0});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process(float32LeBytes(std::vector<float>(100, 0.5F)), output).status,
    fa_agc::backends::ProcessStatus::kOk);

  const fa_agc::backends::ProcessResult release_result =
    backend.process(float32LeBytes(std::vector<float>(100, 0.125F)), output);
  EXPECT_EQ(release_result.status, fa_agc::backends::ProcessStatus::kOk);
  EXPECT_EQ(release_result.gain_direction, fa_agc::backends::GainDirection::kIncrease);
  EXPECT_GT(release_result.committed_gain, 1.0);
  EXPECT_LT(release_result.committed_gain, 2.0);

  const fa_agc::backends::ProcessResult attack_result =
    backend.process(float32LeBytes(std::vector<float>(100, 1.0F)), output);
  EXPECT_EQ(attack_result.status, fa_agc::backends::ProcessStatus::kOk);
  EXPECT_EQ(attack_result.gain_direction, fa_agc::backends::GainDirection::kReduction);
  EXPECT_LT(attack_result.committed_gain, 0.51);
  EXPECT_GT(attack_result.committed_gain, 0.49);
}

TEST(InternalRmsAgcBackendContract, OutputOverflowDoesNotCommitStateOrOverwriteOutput)
{
  fa_agc::backends::InternalRmsAgcBackend backend(
    fa_agc::backends::InternalRmsAgcConfig{10, 1, 0.5, 0.25, 4.0, 1000000.0, 1.0});

  std::vector<uint8_t> output;
  const fa_agc::backends::ProcessResult silence_result =
    backend.process(float32LeBytes(std::vector<float>(10, 0.0F)), output);
  ASSERT_EQ(silence_result.status, fa_agc::backends::ProcessStatus::kOk);
  ASSERT_DOUBLE_EQ(silence_result.target_gain, 4.0);
  ASSERT_DOUBLE_EQ(backend.lastFrameRms(), 0.0);
  ASSERT_DOUBLE_EQ(backend.lastTargetGain(), 4.0);
  const double committed_gain = backend.currentGain();

  output = float32LeBytes({0.125F});
  const fa_agc::backends::ProcessResult overflow_result =
    backend.process(float32LeBytes(std::vector<float>(10, 1.0F)), output);

  EXPECT_EQ(overflow_result.status, fa_agc::backends::ProcessStatus::kOutOfRangeOutput);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.125F}));
  EXPECT_DOUBLE_EQ(backend.currentGain(), committed_gain);
  EXPECT_DOUBLE_EQ(backend.lastFrameRms(), 0.0);
  EXPECT_DOUBLE_EQ(backend.lastTargetGain(), 4.0);
}

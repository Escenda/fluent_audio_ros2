#include "fa_gain/backends/internal_gain.hpp"

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

TEST(InternalGainBackendContract, AppliesLinearGainToCanonicalSamples)
{
  fa_gain::backends::InternalGainBackend backend(
    fa_gain::backends::InternalGainConfig{1, 2.0});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process(float32LeBytes({0.0F, 0.25F, -0.25F}), output),
    fa_gain::backends::ProcessStatus::kOk);

  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.0F, 0.5F, -0.5F}));
}

TEST(InternalGainBackendContract, SupportsInterleavedChannelAlignment)
{
  fa_gain::backends::InternalGainBackend backend(
    fa_gain::backends::InternalGainConfig{2, 0.5});

  std::vector<uint8_t> output;
  ASSERT_EQ(
    backend.process(float32LeBytes({0.5F, -0.5F, 1.0F, -1.0F}), output),
    fa_gain::backends::ProcessStatus::kOk);

  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.25F, -0.25F, 0.5F, -0.5F}));
}

TEST(InternalGainBackendContract, ReportsInputRejectionStatuses)
{
  fa_gain::backends::InternalGainBackend backend(
    fa_gain::backends::InternalGainConfig{1, 1.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output),
    fa_gain::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output),
    fa_gain::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output),
    fa_gain::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output),
    fa_gain::backends::ProcessStatus::kOutOfRangeInput);
}

TEST(InternalGainBackendContract, RejectsOutputOutsideNormalizedRange)
{
  fa_gain::backends::InternalGainBackend backend(
    fa_gain::backends::InternalGainConfig{1, 2.0});

  std::vector<uint8_t> output = float32LeBytes({0.125F});
  EXPECT_EQ(
    backend.process(float32LeBytes({0.75F}), output),
    fa_gain::backends::ProcessStatus::kOutOfRangeOutput);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.125F}));
}

TEST(InternalGainBackendContract, RejectsUnrepresentableOutput)
{
  fa_gain::backends::InternalGainBackend backend(
    fa_gain::backends::InternalGainConfig{1, std::numeric_limits<double>::max()});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process(float32LeBytes({1.0F}), output),
    fa_gain::backends::ProcessStatus::kNonFiniteOutput);
}

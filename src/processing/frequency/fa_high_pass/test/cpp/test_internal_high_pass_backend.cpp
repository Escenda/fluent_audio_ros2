#include "fa_high_pass/backends/internal_high_pass.hpp"

#include <cmath>
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

TEST(InternalHighPassBackendContract, FiltersFloat32LeWithFirstOrderState)
{
  fa_high_pass::backends::InternalHighPassBackend backend(
    fa_high_pass::backends::InternalHighPassConfig{1000, 1, 100.0});

  std::vector<uint8_t> output;
  const auto status = backend.process(float32LeBytes({0.0F, 1.0F, 1.0F}), output);

  ASSERT_EQ(status, fa_high_pass::backends::ProcessStatus::kOk);
  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 3U);
  EXPECT_FLOAT_EQ(samples.at(0), 0.0F);
  EXPECT_NEAR(samples.at(1), static_cast<float>(backend.alpha()), 1.0e-6F);
  EXPECT_NEAR(samples.at(2), static_cast<float>(backend.alpha() * backend.alpha()), 1.0e-6F);
}

TEST(InternalHighPassBackendContract, ReportsInputRejectionStatuses)
{
  fa_high_pass::backends::InternalHighPassBackend backend(
    fa_high_pass::backends::InternalHighPassConfig{16000, 1, 80.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output),
    fa_high_pass::backends::ProcessStatus::kEmptyInput);

  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output),
    fa_high_pass::backends::ProcessStatus::kMisalignedInput);

  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output),
    fa_high_pass::backends::ProcessStatus::kNonFiniteInput);
}

TEST(InternalHighPassBackendContract, RejectsUnhandledStatusValues)
{
  EXPECT_THROW(
    fa_high_pass::backends::processStatusMessage(
      static_cast<fa_high_pass::backends::ProcessStatus>(999)),
    std::logic_error);
}

TEST(InternalHighPassBackendContract, DoesNotCommitStateWhenOutputCannotBeFloat32)
{
  fa_high_pass::backends::InternalHighPassBackend backend(
    fa_high_pass::backends::InternalHighPassConfig{48000, 1, 1.0});

  const float max_sample = std::numeric_limits<float>::max();
  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process(float32LeBytes({-max_sample, max_sample}), output),
    fa_high_pass::backends::ProcessStatus::kNonFiniteOutput);

  ASSERT_EQ(
    backend.process(float32LeBytes({0.0F, 1.0F}), output),
    fa_high_pass::backends::ProcessStatus::kOk);
  const std::vector<float> samples = decodeFloat32Le(output);
  ASSERT_EQ(samples.size(), 2U);
  EXPECT_FLOAT_EQ(samples.at(0), 0.0F);
  EXPECT_NEAR(samples.at(1), static_cast<float>(backend.alpha()), 1.0e-6F);
}

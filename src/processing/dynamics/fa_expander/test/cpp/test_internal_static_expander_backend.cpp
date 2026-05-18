#include "fa_expander/backends/internal_static_expander.hpp"

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

TEST(InternalStaticExpanderBackendContract, RejectsInvalidConfig)
{
  using fa_expander::backends::InternalStaticExpanderBackend;
  using fa_expander::backends::InternalStaticExpanderConfig;

  EXPECT_THROW(
    InternalStaticExpanderBackend(InternalStaticExpanderConfig{0, 0.05, 2.0}),
    std::runtime_error);
  EXPECT_THROW(
    InternalStaticExpanderBackend(InternalStaticExpanderConfig{1, 0.0, 2.0}),
    std::runtime_error);
  EXPECT_THROW(
    InternalStaticExpanderBackend(InternalStaticExpanderConfig{1, 1.0, 2.0}),
    std::runtime_error);
  EXPECT_THROW(
    InternalStaticExpanderBackend(InternalStaticExpanderConfig{1, 0.05, 1.0}),
    std::runtime_error);
}

TEST(InternalStaticExpanderBackendContract, PassesSamplesAtOrAboveThreshold)
{
  fa_expander::backends::InternalStaticExpanderBackend backend(
    fa_expander::backends::InternalStaticExpanderConfig{1, 0.5, 2.0});

  std::vector<uint8_t> output;
  const fa_expander::backends::ProcessResult result =
    backend.process(float32LeBytes({0.5F, -0.5F, 0.75F, -1.0F}), output);

  EXPECT_EQ(result.status, fa_expander::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_expanded, 0U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.5F, -0.5F, 0.75F, -1.0F}));
}

TEST(InternalStaticExpanderBackendContract, ExpandsBelowThresholdAndPreservesSign)
{
  fa_expander::backends::InternalStaticExpanderBackend backend(
    fa_expander::backends::InternalStaticExpanderConfig{1, 0.5, 2.0});

  std::vector<uint8_t> output;
  const fa_expander::backends::ProcessResult result =
    backend.process(float32LeBytes({0.25F, -0.25F, 0.0F, 0.5F}), output);

  EXPECT_EQ(result.status, fa_expander::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_expanded, 3U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.125F, -0.125F, 0.0F, 0.5F}));
}

TEST(InternalStaticExpanderBackendContract, ReportsInputRejectionStatuses)
{
  fa_expander::backends::InternalStaticExpanderBackend backend(
    fa_expander::backends::InternalStaticExpanderConfig{1, 0.5, 2.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output).status,
    fa_expander::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output).status,
    fa_expander::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_expander::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output).status,
    fa_expander::backends::ProcessStatus::kOutOfRangeInput);
}

TEST(InternalStaticExpanderBackendContract, RejectedFrameDoesNotOverwriteOutput)
{
  fa_expander::backends::InternalStaticExpanderBackend backend(
    fa_expander::backends::InternalStaticExpanderConfig{1, 0.5, 2.0});

  std::vector<uint8_t> output = float32LeBytes({0.125F});
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output).status,
    fa_expander::backends::ProcessStatus::kOutOfRangeInput);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.125F}));
}

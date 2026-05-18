#include "fa_dc_offset_removal/backends/internal_frame_mean.hpp"

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

TEST(InternalFrameMeanBackendContract, RejectsInvalidConfig)
{
  EXPECT_THROW(
    fa_dc_offset_removal::backends::InternalFrameMeanBackend(
      fa_dc_offset_removal::backends::InternalFrameMeanConfig{0}),
    std::runtime_error);
}

TEST(InternalFrameMeanBackendContract, RemovesPerChannelFrameMean)
{
  fa_dc_offset_removal::backends::InternalFrameMeanBackend backend(
    fa_dc_offset_removal::backends::InternalFrameMeanConfig{2});

  std::vector<uint8_t> output;
  const fa_dc_offset_removal::backends::ProcessResult result =
    backend.process(float32LeBytes({1.0F, 3.0F, 3.0F, 7.0F}), output);

  EXPECT_EQ(result.status, fa_dc_offset_removal::backends::ProcessStatus::kOk);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{-1.0F, -2.0F, 1.0F, 2.0F}));
}

TEST(InternalFrameMeanBackendContract, PreservesZeroMeanInput)
{
  fa_dc_offset_removal::backends::InternalFrameMeanBackend backend(
    fa_dc_offset_removal::backends::InternalFrameMeanConfig{1});

  std::vector<uint8_t> output;
  const fa_dc_offset_removal::backends::ProcessResult result =
    backend.process(float32LeBytes({-0.5F, 0.5F}), output);

  EXPECT_EQ(result.status, fa_dc_offset_removal::backends::ProcessStatus::kOk);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{-0.5F, 0.5F}));
}

TEST(InternalFrameMeanBackendContract, ReportsInputRejectionStatuses)
{
  fa_dc_offset_removal::backends::InternalFrameMeanBackend backend(
    fa_dc_offset_removal::backends::InternalFrameMeanConfig{1});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output).status,
    fa_dc_offset_removal::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output).status,
    fa_dc_offset_removal::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_dc_offset_removal::backends::ProcessStatus::kNonFiniteInput);
}

TEST(InternalFrameMeanBackendContract, RejectsUnhandledStatusValues)
{
  EXPECT_THROW(
    (void)fa_dc_offset_removal::backends::processStatusMessage(
      static_cast<fa_dc_offset_removal::backends::ProcessStatus>(999)),
    std::logic_error);
}

TEST(InternalFrameMeanBackendContract, RejectedFrameDoesNotOverwriteOutput)
{
  fa_dc_offset_removal::backends::InternalFrameMeanBackend backend(
    fa_dc_offset_removal::backends::InternalFrameMeanConfig{1});

  std::vector<uint8_t> output = float32LeBytes({0.125F});
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_dc_offset_removal::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.125F}));
}

#include "fa_compressor/backends/internal_static_curve.hpp"

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

TEST(InternalStaticCurveBackendContract, PassesSamplesAtOrBelowThreshold)
{
  fa_compressor::backends::InternalStaticCurveBackend backend(
    fa_compressor::backends::InternalStaticCurveConfig{1, 0.5, 4.0, 1.0});

  std::vector<uint8_t> output;
  const fa_compressor::backends::ProcessResult result =
    backend.process(float32LeBytes({0.0F, 0.5F, -0.5F}), output);

  EXPECT_EQ(result.status, fa_compressor::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_compressed, 0U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.0F, 0.5F, -0.5F}));
}

TEST(InternalStaticCurveBackendContract, CompressesAboveThresholdAndPreservesSign)
{
  fa_compressor::backends::InternalStaticCurveBackend backend(
    fa_compressor::backends::InternalStaticCurveConfig{1, 0.5, 4.0, 1.0});

  std::vector<uint8_t> output;
  const fa_compressor::backends::ProcessResult result =
    backend.process(float32LeBytes({0.75F, -0.75F, 0.25F}), output);

  EXPECT_EQ(result.status, fa_compressor::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_compressed, 2U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.5625F, -0.5625F, 0.25F}));
}

TEST(InternalStaticCurveBackendContract, AppliesMakeupGainAfterStaticCurve)
{
  fa_compressor::backends::InternalStaticCurveBackend backend(
    fa_compressor::backends::InternalStaticCurveConfig{1, 0.5, 2.0, 1.5});

  std::vector<uint8_t> output;
  const fa_compressor::backends::ProcessResult result =
    backend.process(float32LeBytes({0.75F}), output);

  EXPECT_EQ(result.status, fa_compressor::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.samples_compressed, 1U);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.9375F}));
}

TEST(InternalStaticCurveBackendContract, ReportsInputAndOutputRejectionStatuses)
{
  fa_compressor::backends::InternalStaticCurveBackend backend(
    fa_compressor::backends::InternalStaticCurveConfig{1, 0.5, 4.0, 2.0});

  std::vector<uint8_t> output;
  EXPECT_EQ(
    backend.process({}, output).status,
    fa_compressor::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0, 1, 2}, output).status,
    fa_compressor::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), output).status,
    fa_compressor::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F}), output).status,
    fa_compressor::backends::ProcessStatus::kOutOfRangeInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({0.75F}), output).status,
    fa_compressor::backends::ProcessStatus::kOutOfRangeOutput);
}

TEST(InternalStaticCurveBackendContract, RejectedFrameDoesNotOverwriteOutput)
{
  fa_compressor::backends::InternalStaticCurveBackend backend(
    fa_compressor::backends::InternalStaticCurveConfig{1, 0.5, 4.0, 2.0});

  std::vector<uint8_t> output = float32LeBytes({0.125F});
  EXPECT_EQ(
    backend.process(float32LeBytes({0.75F}), output).status,
    fa_compressor::backends::ProcessStatus::kOutOfRangeOutput);
  EXPECT_EQ(decodeFloat32Le(output), (std::vector<float>{0.125F}));
}

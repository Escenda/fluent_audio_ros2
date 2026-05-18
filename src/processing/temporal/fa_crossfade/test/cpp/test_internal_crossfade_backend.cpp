#include "fa_crossfade/backends/internal_crossfade.hpp"

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
  bytes.resize(samples.size() * sizeof(float));
  for (size_t i = 0; i < samples.size(); ++i) {
    uint32_t bits = 0U;
    std::memcpy(&bits, &samples[i], sizeof(bits));
    const size_t offset = i * sizeof(float);
    bytes[offset] = static_cast<uint8_t>(bits & 0xFFU);
    bytes[offset + 1U] = static_cast<uint8_t>((bits >> 8U) & 0xFFU);
    bytes[offset + 2U] = static_cast<uint8_t>((bits >> 16U) & 0xFFU);
    bytes[offset + 3U] = static_cast<uint8_t>((bits >> 24U) & 0xFFU);
  }
  return bytes;
}

float readFloat32Le(const std::vector<uint8_t> & bytes, size_t sample_index)
{
  const size_t offset = sample_index * sizeof(float);
  const uint32_t bits =
    static_cast<uint32_t>(bytes[offset]) |
    (static_cast<uint32_t>(bytes[offset + 1U]) << 8U) |
    (static_cast<uint32_t>(bytes[offset + 2U]) << 16U) |
    (static_cast<uint32_t>(bytes[offset + 3U]) << 24U);

  float sample = 0.0F;
  std::memcpy(&sample, &bits, sizeof(sample));
  return sample;
}

}  // namespace

TEST(InternalCrossfadeBackendContract, AppliesLinearCrossfadeBetweenTwoSegments)
{
  fa_crossfade::backends::InternalCrossfadeBackend backend(
    fa_crossfade::backends::InternalCrossfadeConfig{
      1,
      2U,
      fa_crossfade::backends::FadeCurve::kLinear});

  std::vector<uint8_t> output;
  const auto result = backend.process(
    float32LeBytes({0.2F, 0.4F, 0.6F, 0.8F}),
    float32LeBytes({0.1F, 0.3F, 0.5F}),
    output);

  ASSERT_EQ(result.status, fa_crossfade::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.output_frames, 5U);
  ASSERT_EQ(output.size(), 5U * sizeof(float));
  EXPECT_NEAR(readFloat32Le(output, 0), 0.2F, 1.0e-6F);
  EXPECT_NEAR(readFloat32Le(output, 1), 0.4F, 1.0e-6F);
  EXPECT_NEAR(readFloat32Le(output, 2), 0.43333334F, 1.0e-6F);
  EXPECT_NEAR(readFloat32Le(output, 3), 0.46666667F, 1.0e-6F);
  EXPECT_NEAR(readFloat32Le(output, 4), 0.5F, 1.0e-6F);
}

TEST(InternalCrossfadeBackendContract, AppliesEqualPowerCurve)
{
  fa_crossfade::backends::InternalCrossfadeBackend backend(
    fa_crossfade::backends::InternalCrossfadeConfig{
      1,
      1U,
      fa_crossfade::backends::FadeCurve::kEqualPower});

  std::vector<uint8_t> output;
  const auto result = backend.process(float32LeBytes({1.0F}), float32LeBytes({0.0F}), output);

  ASSERT_EQ(result.status, fa_crossfade::backends::ProcessStatus::kOk);
  ASSERT_EQ(output.size(), sizeof(float));
  EXPECT_NEAR(readFloat32Le(output, 0), std::sqrt(0.5F), 1.0e-6F);
}

TEST(InternalCrossfadeBackendContract, ReportsInvalidInputWithoutCommittingOutput)
{
  fa_crossfade::backends::InternalCrossfadeBackend backend(
    fa_crossfade::backends::InternalCrossfadeConfig{
      1,
      2U,
      fa_crossfade::backends::FadeCurve::kLinear});
  std::vector<uint8_t> output = {42U};

  EXPECT_EQ(
    backend.process({}, float32LeBytes({0.0F, 0.0F}), output).status,
    fa_crossfade::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(output, std::vector<uint8_t>{42U});
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{1U, 2U, 3U}, float32LeBytes({0.0F, 0.0F}), output).status,
    fa_crossfade::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({0.0F}), float32LeBytes({0.0F, 0.0F}), output).status,
    fa_crossfade::backends::ProcessStatus::kInputTooShort);
  EXPECT_EQ(
    backend.process(
      float32LeBytes({std::numeric_limits<float>::quiet_NaN(), 0.0F}),
      float32LeBytes({0.0F, 0.0F}),
      output).status,
    fa_crossfade::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F, 0.0F}), float32LeBytes({0.0F, 0.0F}), output).status,
    fa_crossfade::backends::ProcessStatus::kOutOfRangeInput);
  EXPECT_EQ(output, std::vector<uint8_t>{42U});
}

TEST(InternalCrossfadeBackendContract, RejectsOutOfRangeOutputWithoutLimiter)
{
  fa_crossfade::backends::InternalCrossfadeBackend backend(
    fa_crossfade::backends::InternalCrossfadeConfig{
      1,
      1U,
      fa_crossfade::backends::FadeCurve::kEqualPower});
  std::vector<uint8_t> output = {99U};

  const auto result = backend.process(float32LeBytes({1.0F}), float32LeBytes({1.0F}), output);

  EXPECT_EQ(result.status, fa_crossfade::backends::ProcessStatus::kOutOfRangeOutput);
  EXPECT_EQ(output, std::vector<uint8_t>{99U});
}

TEST(InternalCrossfadeBackendContract, RejectsInvalidConfiguration)
{
  EXPECT_THROW(
    fa_crossfade::backends::InternalCrossfadeBackend(
      fa_crossfade::backends::InternalCrossfadeConfig{
        0,
        1U,
        fa_crossfade::backends::FadeCurve::kLinear}),
    std::runtime_error);
  EXPECT_THROW(
    fa_crossfade::backends::InternalCrossfadeBackend(
      fa_crossfade::backends::InternalCrossfadeConfig{
        1,
        0U,
        fa_crossfade::backends::FadeCurve::kLinear}),
    std::runtime_error);
}

TEST(InternalCrossfadeBackendContract, FailsClosedForUnknownNamesAndEnums)
{
  EXPECT_THROW(
    static_cast<void>(fa_crossfade::backends::fadeCurveFromName("unknown")),
    std::logic_error);
  EXPECT_THROW(
    static_cast<void>(
      fa_crossfade::backends::fadeCurveName(static_cast<fa_crossfade::backends::FadeCurve>(999))),
    std::logic_error);
  EXPECT_THROW(
    static_cast<void>(
      fa_crossfade::backends::processStatusMessage(
        static_cast<fa_crossfade::backends::ProcessStatus>(999))),
    std::logic_error);
}

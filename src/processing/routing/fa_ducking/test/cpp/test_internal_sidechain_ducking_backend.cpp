#include "fa_ducking/backends/internal_sidechain_ducking.hpp"

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
  std::vector<uint8_t> bytes(samples.size() * sizeof(float), 0U);
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

fa_ducking::backends::InternalSidechainDuckingBackend makeBackend()
{
  return fa_ducking::backends::InternalSidechainDuckingBackend(
    fa_ducking::backends::InternalSidechainDuckingConfig{
      1,
      1000,
      0.05,
      100000000,
      -12.0,
      10.0,
      250.0});
}

}  // namespace

TEST(InternalSidechainDuckingBackendContract, ObservesSidechainRmsAndDucksProgram)
{
  auto backend = makeBackend();

  const auto sidechain = backend.observeSidechain(float32LeBytes({0.1F, 0.1F}), 1000000);
  ASSERT_EQ(sidechain.status, fa_ducking::backends::ProcessStatus::kOk);
  EXPECT_TRUE(backend.hasSidechain());
  EXPECT_NEAR(sidechain.rms, 0.1, 1.0e-6);

  std::vector<uint8_t> output;
  const auto program = backend.processProgram(float32LeBytes({1.0F, -1.0F}), 2000000, output);

  EXPECT_EQ(program.status, fa_ducking::backends::ProcessStatus::kOk);
  EXPECT_TRUE(program.sidechain_active);
  EXPECT_FALSE(program.sidechain_stale);
  EXPECT_LT(program.output_gain, 1.0);
  ASSERT_EQ(output.size(), 2U * sizeof(float));
  EXPECT_NEAR(readFloat32Le(output, 0), static_cast<float>(program.output_gain), 1.0e-6F);
  EXPECT_NEAR(readFloat32Le(output, 1), static_cast<float>(-program.output_gain), 1.0e-6F);
}

TEST(InternalSidechainDuckingBackendContract, TreatsStaleSidechainAsInactive)
{
  auto backend = makeBackend();
  ASSERT_EQ(
    backend.observeSidechain(float32LeBytes({0.1F}), 1000000).status,
    fa_ducking::backends::ProcessStatus::kOk);

  std::vector<uint8_t> output;
  const auto program = backend.processProgram(float32LeBytes({0.5F}), 200000000, output);

  EXPECT_EQ(program.status, fa_ducking::backends::ProcessStatus::kOk);
  EXPECT_FALSE(program.sidechain_active);
  EXPECT_TRUE(program.sidechain_stale);
  EXPECT_EQ(program.target_gain, 1.0);
}

TEST(InternalSidechainDuckingBackendContract, RejectsInvalidInputWithoutStateCommit)
{
  auto backend = makeBackend();
  ASSERT_EQ(
    backend.observeSidechain(float32LeBytes({0.1F}), 1000000).status,
    fa_ducking::backends::ProcessStatus::kOk);
  std::vector<uint8_t> output = {42U};

  EXPECT_EQ(
    backend.observeSidechain({}, 2000000).status,
    fa_ducking::backends::ProcessStatus::kEmptyInput);
  EXPECT_TRUE(backend.hasSidechain());
  EXPECT_NEAR(backend.lastSidechainRms(), 0.1, 1.0e-6);

  EXPECT_EQ(
    backend.processProgram(float32LeBytes({std::numeric_limits<float>::quiet_NaN()}), 3000000, output).status,
    fa_ducking::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(output, std::vector<uint8_t>{42U});
  EXPECT_EQ(backend.currentGain(), 1.0);
}

TEST(InternalSidechainDuckingBackendContract, RejectsInvalidConfiguration)
{
  EXPECT_THROW(
    fa_ducking::backends::InternalSidechainDuckingBackend(
      fa_ducking::backends::InternalSidechainDuckingConfig{0, 1000, 0.05, 1, -12.0, 10.0, 250.0}),
    std::runtime_error);
  EXPECT_THROW(
    fa_ducking::backends::InternalSidechainDuckingBackend(
      fa_ducking::backends::InternalSidechainDuckingConfig{1, 1000, 0.0, 1, -12.0, 10.0, 250.0}),
    std::runtime_error);
  EXPECT_THROW(
    fa_ducking::backends::InternalSidechainDuckingBackend(
      fa_ducking::backends::InternalSidechainDuckingConfig{1, 1000, 0.05, 1, 0.0, 10.0, 250.0}),
    std::runtime_error);
}

TEST(InternalSidechainDuckingBackendContract, FailsClosedForUnknownStatus)
{
  EXPECT_THROW(
    static_cast<void>(
      fa_ducking::backends::processStatusMessage(
        static_cast<fa_ducking::backends::ProcessStatus>(999))),
    std::logic_error);
}

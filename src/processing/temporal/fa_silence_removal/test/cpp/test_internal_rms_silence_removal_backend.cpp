#include "fa_silence_removal/backends/internal_rms_silence_removal.hpp"

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

}  // namespace

TEST(InternalRmsSilenceRemovalBackendContract, AcceptsActiveFramesAndResetsHangover)
{
  fa_silence_removal::backends::InternalRmsSilenceRemovalBackend backend(
    fa_silence_removal::backends::InternalRmsSilenceRemovalConfig{1, 0.5, 4U});

  const auto result = backend.process(float32LeBytes({1.0F, 0.0F}));

  EXPECT_EQ(result.status, fa_silence_removal::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.decision, fa_silence_removal::backends::Decision::kAcceptedActive);
  EXPECT_NEAR(result.rms, 0.70710678, 1.0e-6);
  EXPECT_EQ(result.frame_count, 2U);
  EXPECT_EQ(result.hangover_samples_remaining, 4U);
  EXPECT_EQ(backend.hangoverSamplesRemaining(), 4U);
}

TEST(InternalRmsSilenceRemovalBackendContract, AcceptsHangoverThenDropsSilentFrames)
{
  fa_silence_removal::backends::InternalRmsSilenceRemovalBackend backend(
    fa_silence_removal::backends::InternalRmsSilenceRemovalConfig{1, 0.5, 2U});

  ASSERT_EQ(
    backend.process(float32LeBytes({1.0F})).decision,
    fa_silence_removal::backends::Decision::kAcceptedActive);

  const auto hangover = backend.process(float32LeBytes({0.0F}));
  EXPECT_EQ(hangover.status, fa_silence_removal::backends::ProcessStatus::kOk);
  EXPECT_EQ(hangover.decision, fa_silence_removal::backends::Decision::kAcceptedHangover);
  EXPECT_EQ(hangover.hangover_samples_remaining, 1U);

  ASSERT_EQ(
    backend.process(float32LeBytes({0.0F})).decision,
    fa_silence_removal::backends::Decision::kAcceptedHangover);

  const auto silent = backend.process(float32LeBytes({0.0F}));
  EXPECT_EQ(silent.status, fa_silence_removal::backends::ProcessStatus::kOk);
  EXPECT_EQ(silent.decision, fa_silence_removal::backends::Decision::kDroppedSilent);
}

TEST(InternalRmsSilenceRemovalBackendContract, ReportsInputRejectionStatusesWithoutCommittingState)
{
  fa_silence_removal::backends::InternalRmsSilenceRemovalBackend backend(
    fa_silence_removal::backends::InternalRmsSilenceRemovalConfig{1, 0.5, 4U});

  ASSERT_EQ(
    backend.process(float32LeBytes({1.0F})).decision,
    fa_silence_removal::backends::Decision::kAcceptedActive);
  ASSERT_NEAR(backend.lastRms(), 1.0, 1.0e-6);
  ASSERT_EQ(backend.hangoverSamplesRemaining(), 4U);

  EXPECT_EQ(
    backend.process({}).status,
    fa_silence_removal::backends::ProcessStatus::kEmptyInput);
  EXPECT_EQ(
    backend.process(std::vector<uint8_t>{0U, 1U, 2U}).status,
    fa_silence_removal::backends::ProcessStatus::kMisalignedInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({std::numeric_limits<float>::quiet_NaN()})).status,
    fa_silence_removal::backends::ProcessStatus::kNonFiniteInput);
  EXPECT_EQ(
    backend.process(float32LeBytes({1.25F})).status,
    fa_silence_removal::backends::ProcessStatus::kOutOfRangeInput);
  EXPECT_NEAR(backend.lastRms(), 1.0, 1.0e-6);
  EXPECT_EQ(backend.hangoverSamplesRemaining(), 4U);
}

TEST(InternalRmsSilenceRemovalBackendContract, RejectsInvalidConfiguration)
{
  EXPECT_THROW(
    fa_silence_removal::backends::InternalRmsSilenceRemovalBackend(
      fa_silence_removal::backends::InternalRmsSilenceRemovalConfig{0, 0.5, 0U}),
    std::runtime_error);
  EXPECT_THROW(
    fa_silence_removal::backends::InternalRmsSilenceRemovalBackend(
      fa_silence_removal::backends::InternalRmsSilenceRemovalConfig{1, -0.1, 0U}),
    std::runtime_error);
  EXPECT_THROW(
    fa_silence_removal::backends::InternalRmsSilenceRemovalBackend(
      fa_silence_removal::backends::InternalRmsSilenceRemovalConfig{1, 1.1, 0U}),
    std::runtime_error);
}

TEST(InternalRmsSilenceRemovalBackendContract, FailsClosedForUnhandledEnumMessages)
{
  EXPECT_THROW(
    static_cast<void>(
      fa_silence_removal::backends::decisionName(
        static_cast<fa_silence_removal::backends::Decision>(999))),
    std::logic_error);
  EXPECT_THROW(
    static_cast<void>(
      fa_silence_removal::backends::processStatusMessage(
        static_cast<fa_silence_removal::backends::ProcessStatus>(999))),
    std::logic_error);
}

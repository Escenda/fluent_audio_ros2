#include "fa_mix/backends/internal_pcm16_mixer.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace
{

void appendPcm16Le(int16_t sample, std::vector<uint8_t> & bytes)
{
  const auto unsigned_sample = static_cast<uint16_t>(sample);
  bytes.push_back(static_cast<uint8_t>(unsigned_sample & 0x00FFU));
  bytes.push_back(static_cast<uint8_t>((unsigned_sample >> 8U) & 0x00FFU));
}

std::vector<uint8_t> pcm16LeBytes(const std::vector<int16_t> & samples)
{
  std::vector<uint8_t> bytes;
  bytes.reserve(samples.size() * sizeof(int16_t));
  for (const int16_t sample : samples) {
    appendPcm16Le(sample, bytes);
  }
  return bytes;
}

fa_mix::backends::InternalPcm16MixerBackend makeBackend()
{
  return fa_mix::backends::InternalPcm16MixerBackend(
    fa_mix::backends::InternalPcm16MixerConfig{1, {0.0, 0.0}});
}

}  // namespace

TEST(InternalPcm16MixerBackendContract, MixesMatchingPcm16Inputs)
{
  auto backend = makeBackend();
  std::vector<uint8_t> output;
  const auto result = backend.mix(
    {pcm16LeBytes({8192, -8192}), pcm16LeBytes({4096, -4096})},
    output);

  EXPECT_EQ(result.status, fa_mix::backends::MixStatus::kOk);
  EXPECT_EQ(result.input_count, 2U);
  EXPECT_EQ(result.sample_count, 2U);
  EXPECT_EQ(output, pcm16LeBytes({12288, -12288}));
  EXPECT_EQ(backend.lastSampleCount(), 2U);
}

TEST(InternalPcm16MixerBackendContract, RejectsSampleCountMismatchWithoutOutputCommit)
{
  auto backend = makeBackend();
  std::vector<uint8_t> output = {42U};
  const auto result = backend.mix(
    {pcm16LeBytes({8192, 8192}), pcm16LeBytes({4096})},
    output);

  EXPECT_EQ(result.status, fa_mix::backends::MixStatus::kSampleCountMismatch);
  EXPECT_EQ(output, std::vector<uint8_t>{42U});
  EXPECT_EQ(backend.lastSampleCount(), 0U);
}

TEST(InternalPcm16MixerBackendContract, RejectsOverflowInsteadOfClamping)
{
  auto backend = makeBackend();
  std::vector<uint8_t> output = {42U};
  const auto result = backend.mix(
    {pcm16LeBytes({32767}), pcm16LeBytes({32767})},
    output);

  EXPECT_EQ(result.status, fa_mix::backends::MixStatus::kOutOfRangeOutput);
  EXPECT_EQ(output, std::vector<uint8_t>{42U});
}

TEST(InternalPcm16MixerBackendContract, RejectsInvalidConfiguration)
{
  EXPECT_THROW(
    fa_mix::backends::InternalPcm16MixerBackend(
      fa_mix::backends::InternalPcm16MixerConfig{0, {0.0}}),
    std::runtime_error);
  EXPECT_THROW(
    fa_mix::backends::InternalPcm16MixerBackend(
      fa_mix::backends::InternalPcm16MixerConfig{1, {}}),
    std::runtime_error);
}

TEST(InternalPcm16MixerBackendContract, FailsClosedForUnknownStatus)
{
  EXPECT_THROW(
    static_cast<void>(
      fa_mix::backends::mixStatusMessage(static_cast<fa_mix::backends::MixStatus>(999))),
    std::logic_error);
}

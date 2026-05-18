#include "fa_monitor_mix/backends/internal_monitor_mix.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace
{
std::vector<uint8_t> floatsToBytes(const std::vector<float> & samples)
{
  std::vector<uint8_t> bytes(samples.size() * sizeof(float));
  std::memcpy(bytes.data(), samples.data(), bytes.size());
  return bytes;
}

std::vector<float> bytesToFloats(const std::vector<uint8_t> & bytes)
{
  std::vector<float> samples(bytes.size() / sizeof(float));
  std::memcpy(samples.data(), bytes.data(), bytes.size());
  return samples;
}

fa_monitor_mix::backends::InternalMonitorMixBackend backendWith(
  const std::vector<double> & gains = {1.0, 1.0})
{
  fa_monitor_mix::backends::InternalMonitorMixConfig config;
  config.input_count = gains.size();
  config.master_index = 0;
  config.channels = 1;
  config.gains_linear = gains;
  return fa_monitor_mix::backends::InternalMonitorMixBackend(config);
}
}  // namespace

TEST(InternalMonitorMixBackendTest, RejectsInvalidConfig)
{
  fa_monitor_mix::backends::InternalMonitorMixConfig config;
  config.input_count = 0;
  config.master_index = 0;
  config.channels = 1;
  config.gains_linear = {};

  EXPECT_THROW(
    (fa_monitor_mix::backends::InternalMonitorMixBackend(config)),
    std::runtime_error);
}

TEST(InternalMonitorMixBackendTest, ValidatesFloat32Range)
{
  auto backend = backendWith();
  EXPECT_EQ(
    backend.validateFrameBytes(floatsToBytes({0.0F, 1.0F, -1.0F})),
    fa_monitor_mix::backends::ProcessStatus::kOk);
  EXPECT_EQ(
    backend.validateFrameBytes(floatsToBytes({1.01F})),
    fa_monitor_mix::backends::ProcessStatus::kOutOfRangeInput);
}

TEST(InternalMonitorMixBackendTest, MixesInputsWithConfiguredGains)
{
  auto backend = backendWith({1.0, 0.5});

  const auto result = backend.mix({
    floatsToBytes({0.25F, -0.25F}),
    floatsToBytes({0.5F, 0.5F}),
  });

  ASSERT_EQ(result.status, fa_monitor_mix::backends::ProcessStatus::kOk);
  const std::vector<float> output = bytesToFloats(result.output);
  ASSERT_EQ(output.size(), 2U);
  EXPECT_FLOAT_EQ(output[0], 0.5F);
  EXPECT_FLOAT_EQ(output[1], 0.0F);
}

TEST(InternalMonitorMixBackendTest, RejectsByteLengthMismatch)
{
  auto backend = backendWith();

  const auto result = backend.mix({
    floatsToBytes({0.0F, 0.0F}),
    floatsToBytes({0.0F}),
  });

  EXPECT_EQ(result.status, fa_monitor_mix::backends::ProcessStatus::kByteLengthMismatch);
  EXPECT_TRUE(result.output.empty());
}

TEST(InternalMonitorMixBackendTest, RejectsOutputRangeOverflowWithoutClamping)
{
  auto backend = backendWith();

  const auto result = backend.mix({
    floatsToBytes({0.75F}),
    floatsToBytes({0.5F}),
  });

  EXPECT_EQ(result.status, fa_monitor_mix::backends::ProcessStatus::kOutOfRangeOutput);
  EXPECT_TRUE(result.output.empty());
}

TEST(InternalMonitorMixBackendTest, RejectsUnhandledStatusValues)
{
  EXPECT_THROW(
    (void)fa_monitor_mix::backends::processStatusMessage(
      static_cast<fa_monitor_mix::backends::ProcessStatus>(999)),
    std::logic_error);
}

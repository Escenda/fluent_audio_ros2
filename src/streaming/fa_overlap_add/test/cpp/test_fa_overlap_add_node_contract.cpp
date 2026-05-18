#include "fa_overlap_add/fa_overlap_add_node.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

namespace
{
using namespace std::chrono_literals;

constexpr const char * kInputTopic = "audio/test/overlap_add_input";
constexpr const char * kOutputTopic = "audio/test/overlap_add_output";
constexpr const char * kInputStreamId = "audio/test/overlap_add_input_stream";
constexpr const char * kOutputStreamId = "audio/test/overlap_add_output_stream";
constexpr uint32_t kSampleRate = 1000;
constexpr uint32_t kChannels = 1;
constexpr uint32_t kBitDepth = 32;
constexpr int64_t kNanosecondsPerSecond = 1000000000LL;

std::vector<rclcpp::Parameter> validParameters()
{
  return {
    rclcpp::Parameter("input_topic", kInputTopic),
    rclcpp::Parameter("output_topic", kOutputTopic),
    rclcpp::Parameter("input_stream_id", kInputStreamId),
    rclcpp::Parameter("output.stream_id", kOutputStreamId),
    rclcpp::Parameter("expected.sample_rate", static_cast<int>(kSampleRate)),
    rclcpp::Parameter("expected.channels", static_cast<int>(kChannels)),
    rclcpp::Parameter("expected.encoding", "FLOAT32LE"),
    rclcpp::Parameter("expected.bit_depth", static_cast<int>(kBitDepth)),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("window.frame_samples", 4),
    rclcpp::Parameter("window.hop_samples", 2),
    rclcpp::Parameter("window.type", "rectangular"),
    rclcpp::Parameter("overlap.max_buffered_chunks", 4),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", true),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
    rclcpp::Parameter("diagnostics.qos.depth", 10),
    rclcpp::Parameter("diagnostics.qos.reliable", true),
  };
}

rclcpp::NodeOptions optionsWith(std::vector<rclcpp::Parameter> parameters)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides(std::move(parameters));
  return options;
}

builtin_interfaces::msg::Time stampFromNanoseconds(const int64_t nanoseconds)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(nanoseconds / kNanosecondsPerSecond);
  stamp.nanosec = static_cast<uint32_t>(nanoseconds % kNanosecondsPerSecond);
  return stamp;
}

int64_t stampToNanoseconds(const builtin_interfaces::msg::Time & stamp)
{
  return (static_cast<int64_t>(stamp.sec) * kNanosecondsPerSecond) +
         static_cast<int64_t>(stamp.nanosec);
}

std::vector<uint8_t> sampleBytes(const std::vector<float> & samples)
{
  std::vector<uint8_t> data(samples.size() * sizeof(float));
  for (size_t index = 0; index < samples.size(); ++index) {
    std::memcpy(data.data() + (index * sizeof(float)), &samples[index], sizeof(float));
  }
  return data;
}

std::vector<float> readSamples(const std::vector<uint8_t> & data)
{
  std::vector<float> samples(data.size() / sizeof(float), 0.0F);
  for (size_t index = 0; index < samples.size(); ++index) {
    std::memcpy(&samples[index], data.data() + (index * sizeof(float)), sizeof(float));
  }
  return samples;
}

fa_interfaces::msg::AudioFrame frameAt(
  const int64_t timestamp_ns,
  const uint32_t epoch,
  const std::vector<float> & samples,
  const std::string & source_id = "mic-a")
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = stampFromNanoseconds(timestamp_ns);
  frame.header.frame_id = source_id + "-frame";
  frame.source_id = source_id;
  frame.stream_id = kInputStreamId;
  frame.encoding = "FLOAT32LE";
  frame.sample_rate = kSampleRate;
  frame.channels = kChannels;
  frame.bit_depth = kBitDepth;
  frame.layout = "interleaved";
  frame.epoch = epoch;
  frame.data = sampleBytes(samples);
  return frame;
}

bool spinUntil(
  rclcpp::executors::SingleThreadedExecutor & executor,
  const std::function<bool()> & predicate)
{
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    executor.spin_some(10ms);
    std::this_thread::sleep_for(10ms);
  }
  return predicate();
}

void spinFor(
  rclcpp::executors::SingleThreadedExecutor & executor,
  const std::chrono::milliseconds duration)
{
  const auto deadline = std::chrono::steady_clock::now() + duration;
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(10ms);
    std::this_thread::sleep_for(10ms);
  }
}

void expectSamplesNear(
  const std::vector<float> & actual,
  const std::vector<float> & expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 0.000001F);
  }
}

class RclcppContractTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if (!rclcpp::ok()) {
      int argc = 0;
      char ** argv = nullptr;
      rclcpp::init(argc, argv);
    }
  }

  void TearDown() override
  {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

TEST_F(RclcppContractTest, ReconstructsHopSequenceFromRectangularOverlappedChunks)
{
  auto node = std::make_shared<fa_overlap_add::FaOverlapAddNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_overlap_add_reconstruct_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic, rclcpp::QoS(10).reliable());
  auto subscription = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(subscription, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&publisher, &subscription]() {
    return publisher->get_subscription_count() > 0 && subscription->get_publisher_count() > 0;
  }));

  publisher->publish(frameAt(1000000000LL, 0U, {0.1F, 0.2F, 0.3F, 0.4F}));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));
  EXPECT_EQ(received[0].stream_id, kOutputStreamId);
  EXPECT_EQ(received[0].epoch, 0U);
  EXPECT_EQ(stampToNanoseconds(received[0].header.stamp), 1000000000LL);
  expectSamplesNear(readSamples(received[0].data), {0.1F, 0.2F});

  publisher->publish(frameAt(1002000000LL, 1U, {0.3F, 0.4F, 0.5F, 0.6F}));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 2u;
  }));
  EXPECT_EQ(received[1].epoch, 1U);
  EXPECT_EQ(stampToNanoseconds(received[1].header.stamp), 1002000000LL);
  expectSamplesNear(readSamples(received[1].data), {0.3F, 0.4F});
}

TEST_F(RclcppContractTest, InvalidFrameDoesNotMutateAccumulatorState)
{
  auto node = std::make_shared<fa_overlap_add::FaOverlapAddNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_overlap_add_invalid_retention_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic, rclcpp::QoS(10).reliable());
  auto subscription = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(subscription, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&publisher, &subscription]() {
    return publisher->get_subscription_count() > 0 && subscription->get_publisher_count() > 0;
  }));

  publisher->publish(frameAt(1000000000LL, 0U, {0.1F, 0.2F, 0.3F, 0.4F}));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  publisher->publish(frameAt(1002000000LL, 1U, {1.5F, 0.4F, 0.5F, 0.6F}));
  spinFor(executor, 100ms);
  EXPECT_EQ(received.size(), 1u);

  publisher->publish(frameAt(1002000000LL, 1U, {0.3F, 0.4F, 0.5F, 0.6F}));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 2u;
  }));
  expectSamplesNear(readSamples(received[1].data), {0.3F, 0.4F});
}

TEST_F(RclcppContractTest, FutureEpochGapResetsStateWithoutMixingOldTail)
{
  auto node = std::make_shared<fa_overlap_add::FaOverlapAddNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_overlap_add_future_gap_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic, rclcpp::QoS(10).reliable());
  auto subscription = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(subscription, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&publisher, &subscription]() {
    return publisher->get_subscription_count() > 0 && subscription->get_publisher_count() > 0;
  }));

  publisher->publish(frameAt(1000000000LL, 0U, {0.1F, 0.2F, 0.3F, 0.4F}));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  publisher->publish(frameAt(2000000000LL, 3U, {0.7F, 0.8F, 0.9F, 1.0F}));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 2u;
  }));
  EXPECT_EQ(received[1].epoch, 1U);
  EXPECT_EQ(stampToNanoseconds(received[1].header.stamp), 2000000000LL);
  expectSamplesNear(readSamples(received[1].data), {0.7F, 0.8F});
}

TEST_F(RclcppContractTest, DuplicateInputEpochDropsWithoutReissuingOutput)
{
  auto node = std::make_shared<fa_overlap_add::FaOverlapAddNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_overlap_add_duplicate_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic, rclcpp::QoS(10).reliable());
  auto subscription = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(subscription, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&publisher, &subscription]() {
    return publisher->get_subscription_count() > 0 && subscription->get_publisher_count() > 0;
  }));

  publisher->publish(frameAt(1000000000LL, 0U, {0.1F, 0.2F, 0.3F, 0.4F}));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  publisher->publish(frameAt(1002000000LL, 0U, {0.3F, 0.4F, 0.5F, 0.6F}));
  spinFor(executor, 200ms);

  EXPECT_EQ(received.size(), 1u);
}
}  // namespace

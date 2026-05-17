#include "fa_latency_compensation/fa_latency_compensation_node.hpp"

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

constexpr const char * kInputTopic = "audio/test/latency_input";
constexpr const char * kOutputTopic = "audio/test/latency_output";
constexpr uint32_t kSampleRate = 16000;
constexpr uint32_t kChannels = 1;
constexpr uint32_t kBitDepth = 32;
constexpr int64_t kNanosecondsPerSecond = 1000000000LL;
constexpr int64_t kMaxBuiltinTimeNanoseconds =
  (2147483647LL * kNanosecondsPerSecond) + 999999999LL;

std::vector<rclcpp::Parameter> validParameters()
{
  return {
    rclcpp::Parameter("input_topic", kInputTopic),
    rclcpp::Parameter("output_topic", kOutputTopic),
    rclcpp::Parameter("compensation.offset_ms", 20.0),
    rclcpp::Parameter("expected.sample_rate", static_cast<int>(kSampleRate)),
    rclcpp::Parameter("expected.channels", static_cast<int>(kChannels)),
    rclcpp::Parameter("expected.encoding", "FLOAT32LE"),
    rclcpp::Parameter("expected.bit_depth", static_cast<int>(kBitDepth)),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", true),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
  };
}

void replaceParameter(
  std::vector<rclcpp::Parameter> & parameters,
  const rclcpp::Parameter & replacement)
{
  for (auto & parameter : parameters) {
    if (parameter.get_name() == replacement.get_name()) {
      parameter = replacement;
      return;
    }
  }
  throw std::logic_error("test parameter replacement target is missing: " + replacement.get_name());
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

std::vector<uint8_t> sampleBytes(const float sample)
{
  std::vector<uint8_t> data(sizeof(float));
  std::memcpy(data.data(), &sample, sizeof(float));
  return data;
}

fa_interfaces::msg::AudioFrame frameAt(
  const int64_t timestamp_ns,
  const uint32_t epoch,
  const float sample = 0.0F)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = stampFromNanoseconds(timestamp_ns);
  frame.header.frame_id = "mic-a-frame";
  frame.source_id = "mic-a";
  frame.stream_id = kInputTopic;
  frame.encoding = "FLOAT32LE";
  frame.sample_rate = kSampleRate;
  frame.channels = kChannels;
  frame.bit_depth = kBitDepth;
  frame.layout = "interleaved";
  frame.epoch = epoch;
  frame.data = sampleBytes(sample);
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

TEST_F(RclcppContractTest, PositiveOffsetUpdatesStampAndPreservesPayloadContract)
{
  auto node = std::make_shared<fa_latency_compensation::FaLatencyCompensationNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_latency_positive_contract_io");
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

  const auto input = frameAt(1000000000LL, 7U, 0.25F);
  publisher->publish(input);
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  EXPECT_EQ(stampToNanoseconds(received[0].header.stamp), 1020000000LL);
  EXPECT_EQ(received[0].stream_id, kOutputTopic);
  EXPECT_EQ(received[0].source_id, input.source_id);
  EXPECT_EQ(received[0].epoch, input.epoch);
  EXPECT_EQ(received[0].data, input.data);
}

TEST_F(RclcppContractTest, NegativeOffsetUpdatesStampWithoutPayloadMutation)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("compensation.offset_ms", -10.0));
  auto node = std::make_shared<fa_latency_compensation::FaLatencyCompensationNode>(
    optionsWith(std::move(parameters)));
  auto io_node = std::make_shared<rclcpp::Node>("fa_latency_negative_contract_io");
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

  const auto input = frameAt(1000000000LL, 8U, -0.5F);
  publisher->publish(input);
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  EXPECT_EQ(stampToNanoseconds(received[0].header.stamp), 990000000LL);
  EXPECT_EQ(received[0].stream_id, kOutputTopic);
  EXPECT_EQ(received[0].data, input.data);
}

TEST_F(RclcppContractTest, DropsNegativeAdjustedTimestamp)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("compensation.offset_ms", -10.0));
  auto node = std::make_shared<fa_latency_compensation::FaLatencyCompensationNode>(
    optionsWith(std::move(parameters)));
  auto io_node = std::make_shared<rclcpp::Node>("fa_latency_negative_drop_contract_io");
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

  publisher->publish(frameAt(5000000LL, 1U));
  spinFor(executor, 200ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, DropsTimestampBeyondBuiltinTimeRange)
{
  auto node = std::make_shared<fa_latency_compensation::FaLatencyCompensationNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_latency_overflow_contract_io");
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

  publisher->publish(frameAt(kMaxBuiltinTimeNanoseconds, 1U));
  spinFor(executor, 200ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, DropsInvalidFormatFramesWithoutPublishing)
{
  auto node = std::make_shared<fa_latency_compensation::FaLatencyCompensationNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_latency_invalid_format_contract_io");
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

  auto invalid = frameAt(1000000000LL, 1U);
  invalid.sample_rate = 48000;
  publisher->publish(invalid);
  spinFor(executor, 200ms);

  EXPECT_TRUE(received.empty());
}
}  // namespace

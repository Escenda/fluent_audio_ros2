#include "fa_clock_drift/fa_clock_drift_node.hpp"

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

rclcpp::NodeOptions quietContractNodeOptions()
{
  rclcpp::NodeOptions options;
  options.start_parameter_services(false);
  options.start_parameter_event_publisher(false);
  options.enable_rosout(false);
  return options;
}

constexpr const char * kInputTopic = "audio/test/clock_drift_input";
constexpr const char * kOutputTopic = "audio/test/clock_drift_output";
constexpr const char * kInputStreamId = "audio/test/clock_drift_input_stream";
constexpr const char * kOutputStreamId = "audio/test/clock_drift_output_stream";
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
    rclcpp::Parameter("drift.ema_alpha", 1.0),
    rclcpp::Parameter("drift.max_correction_ms_per_frame", 0.5),
    rclcpp::Parameter("drift.reset_threshold_ms", 50.0),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", false),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
    rclcpp::Parameter("diagnostics.qos.depth", 10),
    rclcpp::Parameter("diagnostics.qos.reliable", true),
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
  rclcpp::NodeOptions options = quietContractNodeOptions();
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
  const float sample = 0.0F,
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

TEST_F(RclcppContractTest, PublishesBaselineFrameAndPreservesPayloadContract)
{
  auto node = std::make_shared<fa_clock_drift::FaClockDriftNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_clock_drift_baseline_contract_io", quietContractNodeOptions());
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic, rclcpp::QoS(10).best_effort());
  auto subscription = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).best_effort(),
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

  const auto input = frameAt(1000000000LL, 1U, 0.25F);
  publisher->publish(input);
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  EXPECT_EQ(stampToNanoseconds(received[0].header.stamp), 1000000000LL);
  EXPECT_EQ(received[0].stream_id, kOutputStreamId);
  EXPECT_EQ(received[0].source_id, input.source_id);
  EXPECT_EQ(received[0].epoch, input.epoch);
  EXPECT_EQ(received[0].data, input.data);
}

TEST_F(RclcppContractTest, AppliesBoundedCorrectionToSmallObservedDrift)
{
  auto node = std::make_shared<fa_clock_drift::FaClockDriftNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_clock_drift_correction_contract_io", quietContractNodeOptions());
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic, rclcpp::QoS(10).best_effort());
  auto subscription = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).best_effort(),
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

  publisher->publish(frameAt(1000000000LL, 1U, 0.25F));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  const auto second_input = frameAt(1002000000LL, 2U, -0.5F);
  publisher->publish(second_input);
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 2u;
  }));

  EXPECT_EQ(stampToNanoseconds(received[1].header.stamp), 1001500000LL);
  EXPECT_EQ(received[1].stream_id, kOutputStreamId);
  EXPECT_EQ(received[1].source_id, second_input.source_id);
  EXPECT_EQ(received[1].epoch, second_input.epoch);
  EXPECT_EQ(received[1].data, second_input.data);
}

TEST_F(RclcppContractTest, ResetThresholdPublishesNewBaselineTimestamp)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("drift.reset_threshold_ms", 5.0));
  auto node = std::make_shared<fa_clock_drift::FaClockDriftNode>(
    optionsWith(std::move(parameters)));
  auto io_node = std::make_shared<rclcpp::Node>("fa_clock_drift_reset_contract_io", quietContractNodeOptions());
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic, rclcpp::QoS(10).best_effort());
  auto subscription = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).best_effort(),
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

  publisher->publish(frameAt(1000000000LL, 1U));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  publisher->publish(frameAt(2000000000LL, 2U));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 2u;
  }));

  EXPECT_EQ(stampToNanoseconds(received[1].header.stamp), 2000000000LL);
  EXPECT_EQ(received[1].stream_id, kOutputStreamId);
}

TEST_F(RclcppContractTest, SourceChangeStartsNewTimelineWithoutMixingDriftState)
{
  auto node = std::make_shared<fa_clock_drift::FaClockDriftNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_clock_drift_source_switch_contract_io", quietContractNodeOptions());
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic, rclcpp::QoS(10).best_effort());
  auto subscription = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).best_effort(),
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

  publisher->publish(frameAt(1000000000LL, 1U, 0.0F, "mic-a"));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  const auto switched = frameAt(1002000000LL, 2U, 0.125F, "mic-b");
  publisher->publish(switched);
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 2u;
  }));

  EXPECT_EQ(stampToNanoseconds(received[1].header.stamp), 1002000000LL);
  EXPECT_EQ(received[1].source_id, "mic-b");
  EXPECT_EQ(received[1].data, switched.data);
}

TEST_F(RclcppContractTest, DropsInvalidFormatFramesWithoutPublishing)
{
  auto node = std::make_shared<fa_clock_drift::FaClockDriftNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_clock_drift_invalid_format_contract_io", quietContractNodeOptions());
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic, rclcpp::QoS(10).best_effort());
  auto subscription = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).best_effort(),
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
  invalid.encoding = "PCM16LE";
  invalid.bit_depth = 16;
  publisher->publish(invalid);
  spinFor(executor, 200ms);

  EXPECT_TRUE(received.empty());
}
}  // namespace

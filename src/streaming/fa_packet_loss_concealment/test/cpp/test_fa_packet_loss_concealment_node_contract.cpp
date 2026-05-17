#include "fa_packet_loss_concealment/fa_packet_loss_concealment_node.hpp"

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

constexpr const char * kInputTopic = "audio/test/plc_input";
constexpr const char * kOutputTopic = "audio/test/plc_output";
constexpr uint32_t kSampleRate = 1000;
constexpr uint32_t kChannels = 1;
constexpr uint32_t kBitDepth = 32;
constexpr int64_t kNanosecondsPerSecond = 1000000000LL;

std::vector<rclcpp::Parameter> validParameters()
{
  return {
    rclcpp::Parameter("input_topic", kInputTopic),
    rclcpp::Parameter("output_topic", kOutputTopic),
    rclcpp::Parameter("expected.sample_rate", static_cast<int>(kSampleRate)),
    rclcpp::Parameter("expected.channels", static_cast<int>(kChannels)),
    rclcpp::Parameter("expected.encoding", "FLOAT32LE"),
    rclcpp::Parameter("expected.bit_depth", static_cast<int>(kBitDepth)),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("plc.max_gap_frames", 3),
    rclcpp::Parameter("plc.attenuation_per_gap", 0.5),
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

float readFloatSample(const std::vector<uint8_t> & data)
{
  float sample = 0.0F;
  std::memcpy(&sample, data.data(), sizeof(float));
  return sample;
}

fa_interfaces::msg::AudioFrame frameAt(
  const int64_t timestamp_ns,
  const uint32_t epoch,
  const float sample,
  const std::string & source_id = "mic-a")
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = stampFromNanoseconds(timestamp_ns);
  frame.header.frame_id = source_id + "-frame";
  frame.source_id = source_id;
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

TEST_F(RclcppContractTest, SynthesizesMissingEpochsWithAttenuatedPreviousFrame)
{
  auto node = std::make_shared<fa_packet_loss_concealment::FaPacketLossConcealmentNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_plc_gap_contract_io");
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

  publisher->publish(frameAt(1000000000LL, 1U, 0.8F));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  const auto current = frameAt(1003000000LL, 4U, 0.1F);
  publisher->publish(current);
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 4u;
  }));

  EXPECT_EQ(received[0].epoch, 1U);
  EXPECT_EQ(received[1].epoch, 2U);
  EXPECT_EQ(received[2].epoch, 3U);
  EXPECT_EQ(received[3].epoch, 4U);
  EXPECT_EQ(stampToNanoseconds(received[1].header.stamp), 1001000000LL);
  EXPECT_EQ(stampToNanoseconds(received[2].header.stamp), 1002000000LL);
  EXPECT_NEAR(readFloatSample(received[1].data), 0.4F, 0.000001F);
  EXPECT_NEAR(readFloatSample(received[2].data), 0.2F, 0.000001F);
  EXPECT_EQ(received[3].data, current.data);
  for (const auto & frame : received) {
    EXPECT_EQ(frame.stream_id, kOutputTopic);
  }
}

TEST_F(RclcppContractTest, OversizedGapPublishesCurrentFrameOnly)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("plc.max_gap_frames", 1));
  auto node = std::make_shared<fa_packet_loss_concealment::FaPacketLossConcealmentNode>(
    optionsWith(std::move(parameters)));
  auto io_node = std::make_shared<rclcpp::Node>("fa_plc_oversized_gap_contract_io");
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

  publisher->publish(frameAt(1000000000LL, 1U, 0.5F));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  publisher->publish(frameAt(1003000000LL, 4U, 0.125F));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 2u;
  }));

  EXPECT_EQ(received[0].epoch, 1U);
  EXPECT_EQ(received[1].epoch, 4U);
  EXPECT_NEAR(readFloatSample(received[1].data), 0.125F, 0.000001F);
}

TEST_F(RclcppContractTest, DuplicateEpochDropsWithoutPublishing)
{
  auto node = std::make_shared<fa_packet_loss_concealment::FaPacketLossConcealmentNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_plc_duplicate_contract_io");
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

  publisher->publish(frameAt(1000000000LL, 1U, 0.5F));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  publisher->publish(frameAt(1001000000LL, 1U, -0.5F));
  spinFor(executor, 200ms);

  EXPECT_EQ(received.size(), 1u);
}

TEST_F(RclcppContractTest, SourceChangeResetsBaselineAndAvoidsConcealment)
{
  auto node = std::make_shared<fa_packet_loss_concealment::FaPacketLossConcealmentNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_plc_source_switch_contract_io");
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

  publisher->publish(frameAt(1000000000LL, 1U, 0.5F, "mic-a"));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  publisher->publish(frameAt(1002000000LL, 3U, 0.25F, "mic-b"));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 2u;
  }));

  EXPECT_EQ(received[1].epoch, 3U);
  EXPECT_EQ(received[1].source_id, "mic-b");
  EXPECT_NEAR(readFloatSample(received[1].data), 0.25F, 0.000001F);
}

TEST_F(RclcppContractTest, DropsInvalidNormalizedFloatSamplesWithoutPublishing)
{
  auto node = std::make_shared<fa_packet_loss_concealment::FaPacketLossConcealmentNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_plc_invalid_sample_contract_io");
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

  publisher->publish(frameAt(1000000000LL, 1U, 1.5F));
  spinFor(executor, 200ms);

  EXPECT_TRUE(received.empty());
}
}  // namespace

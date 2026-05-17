#include "fa_jitter_buffer/fa_jitter_buffer_node.hpp"

#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

namespace
{
using namespace std::chrono_literals;

constexpr const char * kInputTopic = "audio/test/jitter_input";
constexpr const char * kOutputTopic = "audio/test/jitter_output";
constexpr uint32_t kSampleRate = 16000;
constexpr uint32_t kChannels = 1;
constexpr uint32_t kBitDepth = 32;

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
    rclcpp::Parameter("jitter.target_depth_frames", 1),
    rclcpp::Parameter("jitter.max_depth_frames", 4),
    rclcpp::Parameter("jitter.reset_on_epoch_regression", false),
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

std::vector<uint8_t> sampleBytes(const float sample)
{
  std::vector<uint8_t> data(sizeof(float));
  std::memcpy(data.data(), &sample, sizeof(float));
  return data;
}

fa_interfaces::msg::AudioFrame frameWithEpoch(
  const uint32_t epoch,
  const float sample = 0.0F,
  const std::string & source_id = "network-mic")
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = rclcpp::Clock().now();
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

TEST_F(RclcppContractTest, PublishesEpochOrderedFramesAfterTargetDepth)
{
  auto node = std::make_shared<fa_jitter_buffer::FaJitterBufferNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_jitter_buffer_order_contract_io");
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  publisher->publish(frameWithEpoch(2));
  spinFor(executor, 100ms);
  EXPECT_TRUE(received.empty());

  publisher->publish(frameWithEpoch(1));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));
  EXPECT_EQ(received[0].epoch, 1u);
  EXPECT_EQ(received[0].stream_id, kOutputTopic);

  publisher->publish(frameWithEpoch(3));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 2u;
  }));
  EXPECT_EQ(received[1].epoch, 2u);
}

TEST_F(RclcppContractTest, DropsInvalidFloatSamplesWithoutPublishing)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("jitter.target_depth_frames", 0));
  auto node = std::make_shared<fa_jitter_buffer::FaJitterBufferNode>(
    optionsWith(std::move(parameters)));
  auto io_node = std::make_shared<rclcpp::Node>("fa_jitter_buffer_invalid_contract_io");
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  publisher->publish(frameWithEpoch(1, 1.5F));
  spinFor(executor, 200ms);
  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, DropsDuplicatePublishedEpochs)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("jitter.target_depth_frames", 0));
  auto node = std::make_shared<fa_jitter_buffer::FaJitterBufferNode>(
    optionsWith(std::move(parameters)));
  auto io_node = std::make_shared<rclcpp::Node>("fa_jitter_buffer_duplicate_contract_io");
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  publisher->publish(frameWithEpoch(1));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  publisher->publish(frameWithEpoch(1));
  spinFor(executor, 200ms);
  EXPECT_EQ(received.size(), 1u);
}

TEST_F(RclcppContractTest, SourceContractChangeResetsBufferedFrames)
{
  auto node = std::make_shared<fa_jitter_buffer::FaJitterBufferNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_jitter_buffer_reset_contract_io");
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  publisher->publish(frameWithEpoch(1, 0.0F, "mic-a"));
  const auto replacement = frameWithEpoch(2, 0.0F, "mic-b");
  publisher->publish(replacement);
  spinFor(executor, 100ms);
  EXPECT_TRUE(received.empty());

  publisher->publish(frameWithEpoch(3, 0.0F, "mic-b"));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));
  EXPECT_EQ(received[0].source_id, "mic-b");
  EXPECT_EQ(received[0].epoch, replacement.epoch);
}

}  // namespace

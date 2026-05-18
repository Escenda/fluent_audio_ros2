#include "fa_bit_depth/fa_bit_depth_node.hpp"

#include <chrono>
#include <cstdint>
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

constexpr const char * kInputTopic = "audio/test/bit_depth_input";
constexpr const char * kOutputTopic = "audio/test/bit_depth_output";
constexpr const char * kInputStreamId = "audio/test/bit_depth_input_stream";
constexpr const char * kOutputStreamId = "audio/test/bit_depth_output_stream";
constexpr uint32_t kSampleRate = 16000;
constexpr uint32_t kChannels = 1;
constexpr uint32_t kInputBitDepth = 16;
constexpr uint32_t kOutputBitDepth = 32;
constexpr int64_t kNanosecondsPerSecond = 1000000000LL;

std::vector<rclcpp::Parameter> validParameters()
{
  return {
    rclcpp::Parameter("input_topic", kInputTopic),
    rclcpp::Parameter("output_topic", kOutputTopic),
    rclcpp::Parameter("input_stream_id", kInputStreamId),
    rclcpp::Parameter("input.encoding", "PCM16LE"),
    rclcpp::Parameter("input.bit_depth", static_cast<int>(kInputBitDepth)),
    rclcpp::Parameter("output.stream_id", kOutputStreamId),
    rclcpp::Parameter("output.encoding", "PCM32LE"),
    rclcpp::Parameter("output.bit_depth", static_cast<int>(kOutputBitDepth)),
    rclcpp::Parameter("expected.sample_rate", static_cast<int>(kSampleRate)),
    rclcpp::Parameter("expected.channels", static_cast<int>(kChannels)),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", true),
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

fa_interfaces::msg::AudioFrame frameWith(
  const std::vector<uint8_t> & data,
  const uint32_t epoch = 1U)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = stampFromNanoseconds(1000000000LL);
  frame.header.frame_id = "mic-a-frame";
  frame.source_id = "mic-a";
  frame.stream_id = kInputStreamId;
  frame.encoding = "PCM16LE";
  frame.sample_rate = kSampleRate;
  frame.channels = kChannels;
  frame.bit_depth = kInputBitDepth;
  frame.layout = "interleaved";
  frame.data = data;
  frame.epoch = epoch;
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

TEST_F(RclcppContractTest, ExpandsPcm16LeSamplesIntoPcm32LeHighWords)
{
  auto node = std::make_shared<fa_bit_depth::FaBitDepthNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_bit_depth_expand_contract_io");
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

  const auto input = frameWith({0x00U, 0x00U, 0xffU, 0x7fU, 0x00U, 0x80U, 0xffU, 0xffU});
  publisher->publish(input);
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  EXPECT_EQ(received[0].stream_id, kOutputStreamId);
  EXPECT_EQ(received[0].source_id, input.source_id);
  EXPECT_EQ(received[0].sample_rate, input.sample_rate);
  EXPECT_EQ(received[0].channels, input.channels);
  EXPECT_EQ(received[0].layout, input.layout);
  EXPECT_EQ(received[0].epoch, input.epoch);
  EXPECT_EQ(received[0].encoding, "PCM32LE");
  EXPECT_EQ(received[0].bit_depth, kOutputBitDepth);
  EXPECT_EQ(
    received[0].data,
    std::vector<uint8_t>({
      0x00U, 0x00U, 0x00U, 0x00U,
      0x00U, 0x00U, 0xffU, 0x7fU,
      0x00U, 0x00U, 0x00U, 0x80U,
      0x00U, 0x00U, 0xffU, 0xffU,
    }));
}

TEST_F(RclcppContractTest, DropsMisalignedPayloadWithoutPublishing)
{
  auto node = std::make_shared<fa_bit_depth::FaBitDepthNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_bit_depth_misaligned_contract_io");
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

  publisher->publish(frameWith({0x00U}));
  spinFor(executor, 200ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, DropsMismatchedRuntimeEncodingWithoutPublishing)
{
  auto node = std::make_shared<fa_bit_depth::FaBitDepthNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_bit_depth_encoding_contract_io");
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

  auto invalid = frameWith({0x00U, 0x00U});
  invalid.encoding = "PCM32LE";
  invalid.bit_depth = 32U;
  publisher->publish(invalid);
  spinFor(executor, 200ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, RejectsLossyDownConversionAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("input.encoding", "PCM32LE"));
  replaceParameter(parameters, rclcpp::Parameter("input.bit_depth", 32));
  replaceParameter(parameters, rclcpp::Parameter("output.encoding", "PCM16LE"));
  replaceParameter(parameters, rclcpp::Parameter("output.bit_depth", 16));

  EXPECT_THROW(
    (std::make_shared<fa_bit_depth::FaBitDepthNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}

TEST_F(RclcppContractTest, RejectsEmptyInputStreamIdAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("input_stream_id", ""));

  EXPECT_THROW(
    (std::make_shared<fa_bit_depth::FaBitDepthNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}

TEST_F(RclcppContractTest, RejectsEmptyOutputStreamIdAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("output.stream_id", ""));

  EXPECT_THROW(
    (std::make_shared<fa_bit_depth::FaBitDepthNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}

TEST_F(RclcppContractTest, RejectsInputStreamIdThatCollidesWithTopicAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("input_stream_id", kInputTopic));

  EXPECT_THROW(
    (std::make_shared<fa_bit_depth::FaBitDepthNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}

TEST_F(RclcppContractTest, RejectsOutputStreamIdThatCollidesWithResolvedTopicAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("output_topic", "/audio/test/resolved_output"));
  replaceParameter(parameters, rclcpp::Parameter("output.stream_id", "audio/test/resolved_output"));

  EXPECT_THROW(
    (std::make_shared<fa_bit_depth::FaBitDepthNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}

TEST_F(RclcppContractTest, RejectsEqualInputAndOutputStreamIdsAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("output.stream_id", kInputStreamId));

  EXPECT_THROW(
    (std::make_shared<fa_bit_depth::FaBitDepthNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}

TEST_F(RclcppContractTest, DropsFrameWithTopicNameAsStreamId)
{
  auto node = std::make_shared<fa_bit_depth::FaBitDepthNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_bit_depth_topic_stream_contract_io");
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

  auto invalid = frameWith({0x00U, 0x00U});
  invalid.stream_id = kInputTopic;
  publisher->publish(invalid);
  spinFor(executor, 200ms);

  EXPECT_TRUE(received.empty());
}
}  // namespace

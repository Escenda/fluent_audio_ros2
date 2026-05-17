#include "fa_mix/fa_mix_node.hpp"

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

constexpr const char * kInputTopicA = "audio/test/mix_a";
constexpr const char * kInputTopicB = "audio/test/mix_b";
constexpr const char * kOutputTopic = "audio/test/mix_output";
constexpr uint32_t kSampleRate = 48000;
constexpr uint32_t kChannels = 1;
constexpr uint32_t kBitDepth = 16;
constexpr int64_t kNanosecondsPerSecond = 1000000000LL;

std::vector<rclcpp::Parameter> validParameters()
{
  return {
    rclcpp::Parameter("input_topics", std::vector<std::string>({kInputTopicA, kInputTopicB})),
    rclcpp::Parameter("input_gains_db", std::vector<double>({0.0, 0.0})),
    rclcpp::Parameter("master_index", 0),
    rclcpp::Parameter("output_topic", kOutputTopic),
    rclcpp::Parameter("expected.sample_rate", static_cast<int>(kSampleRate)),
    rclcpp::Parameter("expected.channels", static_cast<int>(kChannels)),
    rclcpp::Parameter("expected.bit_depth", static_cast<int>(kBitDepth)),
    rclcpp::Parameter("expected.encoding", "PCM16LE"),
    rclcpp::Parameter("max_frame_age_ms", 500),
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

void appendPcm16Le(const int16_t sample, std::vector<uint8_t> & bytes)
{
  const auto unsigned_sample = static_cast<uint16_t>(sample);
  bytes.push_back(static_cast<uint8_t>(unsigned_sample & 0x00ffU));
  bytes.push_back(static_cast<uint8_t>((unsigned_sample >> 8U) & 0x00ffU));
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

fa_interfaces::msg::AudioFrame frameWith(
  const std::string & stream_id,
  const std::vector<int16_t> & samples,
  const uint32_t epoch)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = stampFromNanoseconds(1000000000LL + static_cast<int64_t>(epoch));
  frame.header.frame_id = stream_id;
  frame.source_id = "test-source";
  frame.stream_id = stream_id;
  frame.encoding = "PCM16LE";
  frame.sample_rate = kSampleRate;
  frame.channels = kChannels;
  frame.bit_depth = kBitDepth;
  frame.layout = "interleaved";
  frame.data = pcm16LeBytes(samples);
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

TEST_F(RclcppContractTest, DropsWholeMixWhenConfiguredInputIsMissing)
{
  auto node = std::make_shared<fa_mix::FaMixNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_mix_missing_input_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto input_a = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopicA, rclcpp::QoS(10).reliable());
  auto output = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(output, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&input_a, &output]() {
    return input_a->get_subscription_count() > 0 && output->get_publisher_count() > 0;
  }));

  input_a->publish(frameWith(kInputTopicA, {8192, 8192}, 1U));
  spinFor(executor, 250ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, PublishesOnlyAfterAllInputsHaveFreshMatchingFrames)
{
  auto node = std::make_shared<fa_mix::FaMixNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_mix_complete_input_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto input_a = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopicA, rclcpp::QoS(10).reliable());
  auto input_b = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopicB, rclcpp::QoS(10).reliable());
  auto output = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(output, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&input_a, &input_b, &output]() {
    return input_a->get_subscription_count() > 0 &&
      input_b->get_subscription_count() > 0 &&
      output->get_publisher_count() > 0;
  }));

  input_b->publish(frameWith(kInputTopicB, {8192, 8192}, 2U));
  spinFor(executor, 100ms);
  input_a->publish(frameWith(kInputTopicA, {8192, 8192}, 3U));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1U;
  }));

  EXPECT_EQ(received[0].stream_id, kOutputTopic);
  EXPECT_EQ(received[0].source_id, "test-source");
  EXPECT_EQ(received[0].sample_rate, kSampleRate);
  EXPECT_EQ(received[0].channels, kChannels);
  EXPECT_EQ(received[0].bit_depth, kBitDepth);
  EXPECT_EQ(received[0].encoding, "PCM16LE");
  EXPECT_EQ(received[0].layout, "interleaved");
  EXPECT_EQ(received[0].epoch, 3U);
  EXPECT_EQ(received[0].data, pcm16LeBytes({16384, 16384}));
}

TEST_F(RclcppContractTest, DropsWholeMixWhenInputSampleCountDiffers)
{
  auto node = std::make_shared<fa_mix::FaMixNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_mix_size_mismatch_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto input_a = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopicA, rclcpp::QoS(10).reliable());
  auto input_b = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopicB, rclcpp::QoS(10).reliable());
  auto output = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(output, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&input_a, &input_b, &output]() {
    return input_a->get_subscription_count() > 0 &&
      input_b->get_subscription_count() > 0 &&
      output->get_publisher_count() > 0;
  }));

  input_b->publish(frameWith(kInputTopicB, {8192}, 2U));
  spinFor(executor, 100ms);
  input_a->publish(frameWith(kInputTopicA, {8192, 8192}, 3U));
  spinFor(executor, 250ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, RejectsEmptyInputGainsAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("input_gains_db", std::vector<double>()));

  EXPECT_THROW(
    (std::make_shared<fa_mix::FaMixNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}
}  // namespace

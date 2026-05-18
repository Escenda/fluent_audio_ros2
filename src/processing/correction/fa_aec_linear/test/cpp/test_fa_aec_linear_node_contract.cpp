#include "fa_aec_linear/fa_aec_linear_node.hpp"

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

constexpr const char * kMicTopic = "audio/test/aec_mic";
constexpr const char * kRefTopic = "audio/test/aec_ref";
constexpr const char * kOutputTopic = "audio/test/aec_output";
constexpr uint32_t kSampleRate = 16000;
constexpr uint32_t kChannels = 1;
constexpr uint32_t kBitDepth = 16;
constexpr int64_t kNanosecondsPerSecond = 1000000000LL;

std::vector<rclcpp::Parameter> validParameters()
{
  return {
    rclcpp::Parameter("enabled", true),
    rclcpp::Parameter("mic_topic", kMicTopic),
    rclcpp::Parameter("ref_topic", kRefTopic),
    rclcpp::Parameter("output_topic", kOutputTopic),
    rclcpp::Parameter("expected_sample_rate", static_cast<int>(kSampleRate)),
    rclcpp::Parameter("expected_channels", static_cast<int>(kChannels)),
    rclcpp::Parameter("expected.encoding", "PCM16LE"),
    rclcpp::Parameter("expected.bit_depth", static_cast<int>(kBitDepth)),
    rclcpp::Parameter("ref_timeout_ms", 500),
    rclcpp::Parameter("reference_failure_policy", "drop"),
    rclcpp::Parameter("cancel_gain", 1.0),
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
  const uint32_t epoch,
  const int64_t stamp_nanoseconds)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = stampFromNanoseconds(stamp_nanoseconds);
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

fa_interfaces::msg::AudioFrame frameWith(
  const std::string & stream_id,
  const std::vector<int16_t> & samples,
  const uint32_t epoch)
{
  return frameWith(stream_id, samples, epoch, 1000000000LL + static_cast<int64_t>(epoch));
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

TEST_F(RclcppContractTest, RejectsInvalidExpectedFormatAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("expected.encoding", "PCM32LE"));
  replaceParameter(parameters, rclcpp::Parameter("expected.bit_depth", 32));

  EXPECT_THROW(
    (std::make_shared<fa_aec_linear::FaAecLinearNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}

TEST_F(RclcppContractTest, PublishesSubtractedPcm16WhenMicAndReferenceAreBound)
{
  auto node = std::make_shared<fa_aec_linear::FaAecLinearNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_aec_linear_subtract_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto mic_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kMicTopic, rclcpp::QoS(10).reliable());
  auto ref_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kRefTopic, rclcpp::QoS(10).reliable());
  auto output_sub = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(output_sub, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&mic_pub, &ref_pub, &output_sub]() {
    return mic_pub->get_subscription_count() > 0 &&
      ref_pub->get_subscription_count() > 0 &&
      output_sub->get_publisher_count() > 0;
  }));

  ref_pub->publish(frameWith(kRefTopic, {8192, 4096}, 2U));
  spinFor(executor, 100ms);
  mic_pub->publish(frameWith(kMicTopic, {16384, 8192}, 3U));
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
  EXPECT_EQ(received[0].data, pcm16LeBytes({8192, 4096}));
}

TEST_F(RclcppContractTest, DropsMicFrameWhenDisabled)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("enabled", false));
  auto node = std::make_shared<fa_aec_linear::FaAecLinearNode>(optionsWith(std::move(parameters)));
  auto io_node = std::make_shared<rclcpp::Node>("fa_aec_linear_disabled_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto mic_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kMicTopic, rclcpp::QoS(10).reliable());
  auto ref_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kRefTopic, rclcpp::QoS(10).reliable());
  auto output_sub = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(output_sub, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&mic_pub, &ref_pub, &output_sub]() {
    return mic_pub->get_subscription_count() > 0 &&
      ref_pub->get_subscription_count() > 0 &&
      output_sub->get_publisher_count() > 0;
  }));

  ref_pub->publish(frameWith(kRefTopic, {4096}, 2U));
  spinFor(executor, 100ms);
  mic_pub->publish(frameWith(kMicTopic, {8192}, 3U));
  spinFor(executor, 250ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, DropsMicFrameWithMismatchedStreamId)
{
  auto node = std::make_shared<fa_aec_linear::FaAecLinearNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_aec_linear_mic_stream_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto mic_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kMicTopic, rclcpp::QoS(10).reliable());
  auto ref_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kRefTopic, rclcpp::QoS(10).reliable());
  auto output_sub = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(output_sub, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&mic_pub, &ref_pub, &output_sub]() {
    return mic_pub->get_subscription_count() > 0 &&
      ref_pub->get_subscription_count() > 0 &&
      output_sub->get_publisher_count() > 0;
  }));

  ref_pub->publish(frameWith(kRefTopic, {4096}, 2U));
  spinFor(executor, 100ms);
  mic_pub->publish(frameWith("audio/test/wrong_mic_stream", {8192}, 3U));
  spinFor(executor, 250ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, DoesNotCacheReferenceWithMismatchedStreamId)
{
  auto node = std::make_shared<fa_aec_linear::FaAecLinearNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_aec_linear_ref_stream_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto mic_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kMicTopic, rclcpp::QoS(10).reliable());
  auto ref_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kRefTopic, rclcpp::QoS(10).reliable());
  auto output_sub = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(output_sub, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&mic_pub, &ref_pub, &output_sub]() {
    return mic_pub->get_subscription_count() > 0 &&
      ref_pub->get_subscription_count() > 0 &&
      output_sub->get_publisher_count() > 0;
  }));

  ref_pub->publish(frameWith("audio/test/wrong_ref_stream", {4096}, 2U));
  spinFor(executor, 100ms);
  mic_pub->publish(frameWith(kMicTopic, {8192}, 3U));
  spinFor(executor, 250ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, DropsMicFrameWhenReferenceTimestampIsNewerThanMic)
{
  auto node = std::make_shared<fa_aec_linear::FaAecLinearNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_aec_linear_negative_skew_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto mic_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kMicTopic, rclcpp::QoS(10).reliable());
  auto ref_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kRefTopic, rclcpp::QoS(10).reliable());
  auto output_sub = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(output_sub, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&mic_pub, &ref_pub, &output_sub]() {
    return mic_pub->get_subscription_count() > 0 &&
      ref_pub->get_subscription_count() > 0 &&
      output_sub->get_publisher_count() > 0;
  }));

  ref_pub->publish(frameWith(kRefTopic, {4096}, 2U, 2000000000LL));
  spinFor(executor, 100ms);
  mic_pub->publish(frameWith(kMicTopic, {8192}, 3U, 1500000000LL));
  spinFor(executor, 250ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, DropsMicFrameWhenReferenceTimestampIsTooOld)
{
  auto node = std::make_shared<fa_aec_linear::FaAecLinearNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_aec_linear_stale_ref_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto mic_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kMicTopic, rclcpp::QoS(10).reliable());
  auto ref_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kRefTopic, rclcpp::QoS(10).reliable());
  auto output_sub = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(output_sub, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&mic_pub, &ref_pub, &output_sub]() {
    return mic_pub->get_subscription_count() > 0 &&
      ref_pub->get_subscription_count() > 0 &&
      output_sub->get_publisher_count() > 0;
  }));

  ref_pub->publish(frameWith(kRefTopic, {4096}, 2U, 1000000000LL));
  spinFor(executor, 100ms);
  mic_pub->publish(frameWith(kMicTopic, {8192}, 3U, 1700000000LL));
  spinFor(executor, 250ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, RejectsDisabledChannelValidationAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("expected_channels", -1));

  EXPECT_THROW(
    (std::make_shared<fa_aec_linear::FaAecLinearNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}
}  // namespace

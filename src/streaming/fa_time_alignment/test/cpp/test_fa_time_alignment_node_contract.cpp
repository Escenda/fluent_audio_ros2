#include "fa_time_alignment/fa_time_alignment_node.hpp"

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

constexpr const char * kInputTopic = "audio/test/time_alignment_input";
constexpr const char * kOutputTopic = "audio/test/time_alignment_output";
constexpr const char * kInputStreamId = "audio/test/time_alignment_input_stream";
constexpr const char * kOutputStreamId = "audio/test/time_alignment_output_stream";
constexpr uint32_t kSampleRate = 16000;
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
    rclcpp::Parameter("alignment.period_ms", 20.0),
    rclcpp::Parameter("alignment.phase_ms", 0.0),
    rclcpp::Parameter("alignment.max_adjust_ms", 2.0),
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
  const float sample = 0.0F)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = stampFromNanoseconds(timestamp_ns);
  frame.header.frame_id = "mic-a-frame";
  frame.source_id = "mic-a";
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

TEST_F(RclcppContractTest, AlignsNearestGridAndPreservesAudioPayload)
{
  auto node = std::make_shared<fa_time_alignment::FaTimeAlignmentNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_time_alignment_grid_contract_io", quietContractNodeOptions());
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

  const auto input = frameAt(1001000000LL, 7U, 0.25F);
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

TEST_F(RclcppContractTest, AppliesConfiguredPhaseToGridAlignment)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("alignment.phase_ms", 5.0));
  auto node = std::make_shared<fa_time_alignment::FaTimeAlignmentNode>(
    optionsWith(std::move(parameters)));
  auto io_node = std::make_shared<rclcpp::Node>("fa_time_alignment_phase_contract_io", quietContractNodeOptions());
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

  publisher->publish(frameAt(1006000000LL, 8U, -0.5F));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));

  EXPECT_EQ(stampToNanoseconds(received[0].header.stamp), 1005000000LL);
  EXPECT_EQ(received[0].stream_id, kOutputStreamId);
}

TEST_F(RclcppContractTest, DropsExcessTimestampAdjustment)
{
  auto node = std::make_shared<fa_time_alignment::FaTimeAlignmentNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_time_alignment_excess_contract_io", quietContractNodeOptions());
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

  publisher->publish(frameAt(1010000000LL, 1U, 0.1F));
  spinFor(executor, 200ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, DropsInvalidFormatWithoutPublishing)
{
  auto node = std::make_shared<fa_time_alignment::FaTimeAlignmentNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_time_alignment_invalid_format_contract_io", quietContractNodeOptions());
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

  auto invalid = frameAt(1000000000LL, 1U, 0.1F);
  invalid.channels = 2U;
  publisher->publish(invalid);
  spinFor(executor, 200ms);

  EXPECT_TRUE(received.empty());
}
}  // namespace

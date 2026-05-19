#include "fa_frame_buffer/fa_frame_buffer_node.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
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

rclcpp::NodeOptions quietContractNodeOptions()
{
  rclcpp::NodeOptions options;
  options.start_parameter_services(false);
  options.start_parameter_event_publisher(false);
  options.enable_rosout(false);
  return options;
}

constexpr const char * kInputTopic = "audio/test/input";
constexpr const char * kOutputTopic = "audio/test/output";
constexpr const char * kInputStreamId = "audio/test/input_stream";
constexpr const char * kOutputStreamId = "audio/test/output_stream";
constexpr uint32_t kSampleRate = 16000;
constexpr uint32_t kChannels = 1;
constexpr uint32_t kBitDepth = 32;
constexpr size_t kBytesPerFrame = 4;
constexpr size_t kFramesPerChunk = 4;

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
    rclcpp::Parameter("buffering.frames_per_chunk", static_cast<int>(kFramesPerChunk)),
    rclcpp::Parameter("buffering.max_buffered_chunks", 2),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", false),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
    rclcpp::Parameter("diagnostics.qos.depth", 10),
    rclcpp::Parameter("diagnostics.qos.reliable", true),
  };
}

rclcpp::NodeOptions optionsWith(std::vector<rclcpp::Parameter> parameters)
{
  rclcpp::NodeOptions options = quietContractNodeOptions();
  options.parameter_overrides(std::move(parameters));
  return options;
}

std::vector<uint8_t> bytesForFrames(const size_t frame_count, const uint8_t base)
{
  std::vector<uint8_t> data(frame_count * kBytesPerFrame);
  for (size_t index = 0; index < data.size(); ++index) {
    data[index] = static_cast<uint8_t>(base + index);
  }
  return data;
}

fa_interfaces::msg::AudioFrame frameWithFrames(
  const size_t frame_count,
  const uint8_t base,
  const std::string & source_id = "test-mic")
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = rclcpp::Clock().now();
  frame.header.frame_id = source_id + "-frame";
  frame.source_id = source_id;
  frame.stream_id = kInputStreamId;
  frame.encoding = "FLOAT32LE";
  frame.sample_rate = kSampleRate;
  frame.channels = kChannels;
  frame.bit_depth = kBitDepth;
  frame.layout = "interleaved";
  frame.epoch = 1;
  frame.data = bytesForFrames(frame_count, base);
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

TEST_F(RclcppContractTest, DoesNotPublishPartialChunks)
{
  auto node = std::make_shared<fa_frame_buffer::FaFrameBufferNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_frame_buffer_partial_contract_io", quietContractNodeOptions());
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  publisher->publish(frameWithFrames(2, 0x10));
  spinFor(executor, 200ms);
  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, PublishesOneChunkWhenAccumulatedFramesReachConfiguredSize)
{
  auto node = std::make_shared<fa_frame_buffer::FaFrameBufferNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_frame_buffer_chunk_contract_io", quietContractNodeOptions());
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  const auto first = frameWithFrames(2, 0x20);
  const auto second = frameWithFrames(2, 0x40);
  std::vector<uint8_t> expected = first.data;
  expected.insert(expected.end(), second.data.begin(), second.data.end());

  publisher->publish(first);
  publisher->publish(second);

  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));
  EXPECT_EQ(received[0].stream_id, kOutputStreamId);
  EXPECT_EQ(received[0].source_id, "test-mic");
  EXPECT_EQ(received[0].encoding, "FLOAT32LE");
  EXPECT_EQ(received[0].sample_rate, kSampleRate);
  EXPECT_EQ(received[0].channels, kChannels);
  EXPECT_EQ(received[0].bit_depth, kBitDepth);
  EXPECT_EQ(received[0].layout, "interleaved");
  EXPECT_EQ(received[0].data, expected);
}

TEST_F(RclcppContractTest, PublishesTwoChunksFromDoubleSizedInput)
{
  auto node = std::make_shared<fa_frame_buffer::FaFrameBufferNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_frame_buffer_double_contract_io", quietContractNodeOptions());
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  publisher->publish(frameWithFrames(8, 0x60));

  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 2u;
  }));
  EXPECT_EQ(received[0].data.size(), kFramesPerChunk * kBytesPerFrame);
  EXPECT_EQ(received[1].data.size(), kFramesPerChunk * kBytesPerFrame);
}

TEST_F(RclcppContractTest, DropsInvalidFormatFramesWithoutPublishing)
{
  auto node = std::make_shared<fa_frame_buffer::FaFrameBufferNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_frame_buffer_invalid_contract_io", quietContractNodeOptions());
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  auto invalid = frameWithFrames(4, 0x80);
  invalid.encoding = "PCM16LE";
  invalid.bit_depth = 16;
  publisher->publish(invalid);
  spinFor(executor, 200ms);
  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, ClearsPartialBufferOnStreamIdentityChange)
{
  auto node = std::make_shared<fa_frame_buffer::FaFrameBufferNode>(
    optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_frame_buffer_identity_contract_io", quietContractNodeOptions());
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  publisher->publish(frameWithFrames(2, 0xA0, "mic-a"));
  const auto replacement = frameWithFrames(4, 0xC0, "mic-b");
  publisher->publish(replacement);

  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1u;
  }));
  EXPECT_EQ(received[0].source_id, "mic-b");
  EXPECT_EQ(received[0].data, replacement.data);
}

}  // namespace

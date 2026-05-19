#include "fa_crossfade/fa_crossfade_node.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

namespace
{

using namespace std::chrono_literals;

rclcpp::NodeOptions quietGraphNodeOptions()
{
  rclcpp::NodeOptions options;
  options.enable_rosout(false);
  options.start_parameter_services(false);
  options.start_parameter_event_publisher(false);
  return options;
}

std::vector<uint8_t> float32LeBytes(const std::vector<float> & samples)
{
  std::vector<uint8_t> bytes;
  bytes.resize(samples.size() * sizeof(float));
  for (size_t i = 0; i < samples.size(); ++i) {
    uint32_t bits = 0U;
    std::memcpy(&bits, &samples[i], sizeof(bits));
    const size_t offset = i * sizeof(float);
    bytes[offset] = static_cast<uint8_t>(bits & 0xFFU);
    bytes[offset + 1U] = static_cast<uint8_t>((bits >> 8U) & 0xFFU);
    bytes[offset + 2U] = static_cast<uint8_t>((bits >> 16U) & 0xFFU);
    bytes[offset + 3U] = static_cast<uint8_t>((bits >> 24U) & 0xFFU);
  }
  return bytes;
}

float readFloat32Le(const std::vector<uint8_t> & bytes, size_t index)
{
  const size_t offset = index * sizeof(float);
  const uint32_t bits =
    static_cast<uint32_t>(bytes[offset]) |
    (static_cast<uint32_t>(bytes[offset + 1U]) << 8U) |
    (static_cast<uint32_t>(bytes[offset + 2U]) << 16U) |
    (static_cast<uint32_t>(bytes[offset + 3U]) << 24U);
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

fa_interfaces::msg::AudioFrame makeSegment(
  const rclcpp::Node & node,
  const std::string & stream_id,
  const std::vector<float> & samples,
  uint32_t epoch)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = node.now();
  frame.source_id = "test-mic";
  frame.stream_id = stream_id;
  frame.encoding = "FLOAT32LE";
  frame.sample_rate = 1000;
  frame.channels = 1;
  frame.bit_depth = 32;
  frame.layout = "interleaved";
  frame.data = float32LeBytes(samples);
  frame.epoch = epoch;
  return frame;
}

class RclcppFixture : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    if (!rclcpp::ok()) {
      int argc = 0;
      char ** argv = nullptr;
      rclcpp::init(argc, argv);
    }
  }

  static void TearDownTestSuite()
  {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

}  // namespace

TEST_F(RclcppFixture, PublishesOnlyMatchingEpochPairWithOutputStreamIdentity)
{
  rclcpp::NodeOptions options = quietGraphNodeOptions();
  options.parameter_overrides({
    rclcpp::Parameter("input_a_topic", "/fa_crossfade_test/input_a"),
    rclcpp::Parameter("input_b_topic", "/fa_crossfade_test/input_b"),
    rclcpp::Parameter("output_topic", "/fa_crossfade_test/output"),
    rclcpp::Parameter("input_a_stream_id", "fa_crossfade_test/a_stream"),
    rclcpp::Parameter("input_b_stream_id", "fa_crossfade_test/b_stream"),
    rclcpp::Parameter("output.stream_id", "fa_crossfade_test/output_stream"),
    rclcpp::Parameter("crossfade.overlap_frames", 1),
    rclcpp::Parameter("crossfade.curve", "linear"),
    rclcpp::Parameter("expected.sample_rate", 1000),
    rclcpp::Parameter("expected.channels", 1),
    rclcpp::Parameter("expected.encoding", "FLOAT32LE"),
    rclcpp::Parameter("expected.bit_depth", 32),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", false),
    rclcpp::Parameter("diagnostics.qos.depth", 10),
    rclcpp::Parameter("diagnostics.qos.reliable", false),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
  });

  auto crossfade_node = std::make_shared<fa_crossfade::FaCrossfadeNode>(options);
  auto test_node = std::make_shared<rclcpp::Node>("fa_crossfade_graph_test", quietGraphNodeOptions());

  rclcpp::QoS qos(10);
  qos.best_effort();
  auto publisher_a = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_crossfade_test/input_a",
    qos);
  auto publisher_b = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_crossfade_test/input_b",
    qos);
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_crossfade_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(crossfade_node);
  executor.add_node(test_node);

  auto deadline = std::chrono::steady_clock::now() + 3s;
  while ((publisher_a->get_subscription_count() == 0U ||
          publisher_b->get_subscription_count() == 0U ||
          subscriber->get_publisher_count() == 0U) &&
         std::chrono::steady_clock::now() < deadline)
  {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  publisher_a->publish(makeSegment(*test_node, "wrong_stream", {0.2F, 0.6F}, 100U));
  publisher_b->publish(makeSegment(*test_node, "fa_crossfade_test/b_stream", {0.4F, 0.8F}, 100U));
  deadline = std::chrono::steady_clock::now() + 300ms;
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_TRUE(received.empty());

  publisher_a->publish(makeSegment(*test_node, "fa_crossfade_test/a_stream", {0.2F, 0.6F}, 101U));
  deadline = std::chrono::steady_clock::now() + 300ms;
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_TRUE(received.empty());

  publisher_a->publish(makeSegment(*test_node, "fa_crossfade_test/a_stream", {0.2F, 0.6F}, 7U));
  publisher_b->publish(makeSegment(*test_node, "fa_crossfade_test/b_stream", {0.4F, 0.8F}, 7U));
  deadline = std::chrono::steady_clock::now() + 3s;
  while (received.empty() && std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(crossfade_node);
  subscriber.reset();
  publisher_b.reset();
  publisher_a.reset();

  ASSERT_EQ(received.size(), 1U);
  EXPECT_EQ(received[0].source_id, "test-mic");
  EXPECT_EQ(received[0].stream_id, "fa_crossfade_test/output_stream");
  EXPECT_EQ(received[0].sample_rate, 1000U);
  EXPECT_EQ(received[0].channels, 1U);
  EXPECT_EQ(received[0].encoding, "FLOAT32LE");
  EXPECT_EQ(received[0].bit_depth, 32U);
  EXPECT_EQ(received[0].layout, "interleaved");
  EXPECT_EQ(received[0].epoch, 7U);
  ASSERT_EQ(received[0].data.size(), 3U * sizeof(float));
  EXPECT_FLOAT_EQ(readFloat32Le(received[0].data, 0), 0.2F);
  EXPECT_FLOAT_EQ(readFloat32Le(received[0].data, 1), 0.5F);
  EXPECT_FLOAT_EQ(readFloat32Le(received[0].data, 2), 0.8F);
}

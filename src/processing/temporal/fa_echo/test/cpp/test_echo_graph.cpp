#include "fa_echo/fa_echo_node.hpp"

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

fa_interfaces::msg::AudioFrame makeFloat32Frame(
  const rclcpp::Node & node,
  float sample,
  uint64_t epoch)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = node.now();
  frame.source_id = "test-mic";
  frame.stream_id = "fa_echo_test/input_stream";
  frame.encoding = "FLOAT32LE";
  frame.sample_rate = 1000;
  frame.channels = 1;
  frame.bit_depth = 32;
  frame.layout = "interleaved";
  frame.data = float32LeBytes({sample});
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

TEST_F(RclcppFixture, PublishesEchoFrameWithSeparateStreamIdentity)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("input_topic", "/fa_echo_test/input"),
    rclcpp::Parameter("output_topic", "/fa_echo_test/output"),
    rclcpp::Parameter("input_stream_id", "fa_echo_test/input_stream"),
    rclcpp::Parameter("output.stream_id", "fa_echo_test/output_stream"),
    rclcpp::Parameter("echo.delay_ms", 1.0),
    rclcpp::Parameter("echo.feedback_gain", 0.0),
    rclcpp::Parameter("echo.wet_gain", 0.5),
    rclcpp::Parameter("echo.dry_gain", 1.0),
    rclcpp::Parameter("expected.sample_rate", 1000),
    rclcpp::Parameter("expected.channels", 1),
    rclcpp::Parameter("expected.encoding", "FLOAT32LE"),
    rclcpp::Parameter("expected.bit_depth", 32),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", true),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
  });

  auto echo_node = std::make_shared<fa_echo::FaEchoNode>(options);
  auto test_node = std::make_shared<rclcpp::Node>("fa_echo_graph_test");

  rclcpp::QoS qos(10);
  qos.reliable();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_echo_test/input",
    qos);
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_echo_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(echo_node);
  executor.add_node(test_node);

  auto deadline = std::chrono::steady_clock::now() + 3s;
  while ((publisher->get_subscription_count() == 0U || subscriber->get_publisher_count() == 0U) &&
         std::chrono::steady_clock::now() < deadline)
  {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  deadline = std::chrono::steady_clock::now() + 3s;
  publisher->publish(makeFloat32Frame(*test_node, 1.0F, 1U));
  while (received.size() < 1U && std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  deadline = std::chrono::steady_clock::now() + 3s;
  publisher->publish(makeFloat32Frame(*test_node, 0.0F, 2U));
  while (received.size() < 2U && std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(echo_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_GE(received.size(), 2U);
  EXPECT_EQ(received[0].source_id, "test-mic");
  EXPECT_EQ(received[0].stream_id, "fa_echo_test/output_stream");
  EXPECT_EQ(received[0].sample_rate, 1000U);
  EXPECT_EQ(received[0].channels, 1U);
  EXPECT_EQ(received[0].encoding, "FLOAT32LE");
  EXPECT_EQ(received[0].bit_depth, 32U);
  EXPECT_EQ(received[0].layout, "interleaved");
  EXPECT_EQ(received[0].epoch, 1U);
  ASSERT_EQ(received[0].data.size(), sizeof(float));
  EXPECT_FLOAT_EQ(readFloat32Le(received[0].data, 0), 1.0F);

  EXPECT_EQ(received[1].source_id, "test-mic");
  EXPECT_EQ(received[1].stream_id, "fa_echo_test/output_stream");
  EXPECT_EQ(received[1].epoch, 2U);
  ASSERT_EQ(received[1].data.size(), sizeof(float));
  EXPECT_FLOAT_EQ(readFloat32Le(received[1].data, 0), 0.5F);
}

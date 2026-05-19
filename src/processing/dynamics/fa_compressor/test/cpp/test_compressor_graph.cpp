#include "fa_compressor/fa_compressor_node.hpp"

#include <chrono>
#include <cstring>
#include <memory>
#include <optional>
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
  bytes.reserve(samples.size() * sizeof(float));
  for (const float sample : samples) {
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(float));
    std::memcpy(bytes.data() + offset, &sample, sizeof(float));
  }
  return bytes;
}

float readFloat32Le(const std::vector<uint8_t> & bytes, const size_t index)
{
  const size_t offset = index * sizeof(float);
  float value = 0.0F;
  std::memcpy(&value, bytes.data() + offset, sizeof(float));
  return value;
}

fa_interfaces::msg::AudioFrame makeFloat32Frame(
  const rclcpp::Node & node,
  const std::string & stream_id,
  const uint32_t epoch)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = node.now();
  frame.source_id = "test-mic";
  frame.stream_id = stream_id;
  frame.encoding = "FLOAT32LE";
  frame.sample_rate = 16000;
  frame.channels = 1;
  frame.bit_depth = 32;
  frame.layout = "interleaved";
  frame.data = float32LeBytes({0.0F, 0.25F, 0.75F, -1.0F});
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

TEST_F(RclcppFixture, PublishesCompressedFloat32Frame)
{
  rclcpp::NodeOptions options = quietGraphNodeOptions();
  options.parameter_overrides({
    rclcpp::Parameter("input_topic", "/fa_compressor_test/input"),
    rclcpp::Parameter("output_topic", "/fa_compressor_test/output"),
    rclcpp::Parameter("input_stream_id", "fa_compressor_test/input_stream"),
    rclcpp::Parameter("output.stream_id", "fa_compressor_test/output_stream"),
    rclcpp::Parameter("compressor.threshold_linear", 0.5),
    rclcpp::Parameter("compressor.ratio", 4.0),
    rclcpp::Parameter("compressor.makeup_gain_linear", 1.0),
    rclcpp::Parameter("expected.sample_rate", 16000),
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

  auto compressor_node = std::make_shared<fa_compressor::FaCompressorNode>(options);
  auto test_node = std::make_shared<rclcpp::Node>("fa_compressor_graph_test", quietGraphNodeOptions());

  rclcpp::QoS qos(10);
  qos.best_effort();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_compressor_test/input",
    qos);
  std::optional<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_compressor_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received = *msg;
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(compressor_node);
  executor.add_node(test_node);

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (!received.has_value() && std::chrono::steady_clock::now() < deadline) {
    publisher->publish(makeFloat32Frame(*test_node, "fa_compressor_test/input_stream", 41));
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(compressor_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->source_id, "test-mic");
  EXPECT_EQ(received->stream_id, "fa_compressor_test/output_stream");
  EXPECT_EQ(received->encoding, "FLOAT32LE");
  EXPECT_EQ(received->sample_rate, 16000U);
  EXPECT_EQ(received->channels, 1U);
  EXPECT_EQ(received->bit_depth, 32U);
  EXPECT_EQ(received->layout, "interleaved");
  EXPECT_EQ(received->epoch, 41U);
  ASSERT_EQ(received->data.size(), 4U * sizeof(float));
  EXPECT_FLOAT_EQ(readFloat32Le(received->data, 0), 0.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(received->data, 1), 0.25F);
  EXPECT_FLOAT_EQ(readFloat32Le(received->data, 2), 0.5625F);
  EXPECT_FLOAT_EQ(readFloat32Le(received->data, 3), -0.625F);
}

TEST_F(RclcppFixture, DropsFrameWhenStreamIdDoesNotMatchInputTopic)
{
  rclcpp::NodeOptions options = quietGraphNodeOptions();
  options.parameter_overrides({
    rclcpp::Parameter("input_topic", "/fa_compressor_drop_test/input"),
    rclcpp::Parameter("output_topic", "/fa_compressor_drop_test/output"),
    rclcpp::Parameter("input_stream_id", "fa_compressor_drop_test/input_stream"),
    rclcpp::Parameter("output.stream_id", "fa_compressor_drop_test/output_stream"),
    rclcpp::Parameter("compressor.threshold_linear", 0.5),
    rclcpp::Parameter("compressor.ratio", 4.0),
    rclcpp::Parameter("compressor.makeup_gain_linear", 1.0),
    rclcpp::Parameter("expected.sample_rate", 16000),
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

  auto compressor_node = std::make_shared<fa_compressor::FaCompressorNode>(options);
  auto test_node = std::make_shared<rclcpp::Node>("fa_compressor_drop_graph_test", quietGraphNodeOptions());

  rclcpp::QoS qos(10);
  qos.best_effort();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_compressor_drop_test/input",
    qos);
  std::optional<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_compressor_drop_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received = *msg;
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(compressor_node);
  executor.add_node(test_node);

  auto deadline = std::chrono::steady_clock::now() + 500ms;
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  deadline = std::chrono::steady_clock::now() + 500ms;
  while (std::chrono::steady_clock::now() < deadline) {
    publisher->publish(
      makeFloat32Frame(*test_node, "fa_compressor_drop_test/wrong_input", 42));
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  deadline = std::chrono::steady_clock::now() + 500ms;
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(compressor_node);
  subscriber.reset();
  publisher.reset();

  EXPECT_FALSE(received.has_value());
}

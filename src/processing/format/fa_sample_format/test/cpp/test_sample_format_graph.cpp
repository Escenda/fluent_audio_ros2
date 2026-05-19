#include "fa_sample_format/fa_sample_format_node.hpp"

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

float readFloat32Le(const std::vector<uint8_t> & bytes, const size_t index)
{
  const size_t offset = index * sizeof(float);
  float value = 0.0F;
  std::memcpy(&value, bytes.data() + offset, sizeof(float));
  return value;
}

fa_interfaces::msg::AudioFrame makePcm16Frame(const rclcpp::Node & node)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = node.now();
  frame.source_id = "test-mic";
  frame.stream_id = "audio/test/raw";
  frame.encoding = "PCM16LE";
  frame.sample_rate = 16000;
  frame.channels = 1;
  frame.bit_depth = 16;
  frame.layout = "interleaved";
  frame.data = {
    0x00, 0x80,  // -32768
    0x00, 0x00,  // 0
    0xFF, 0x7F,  // 32767
  };
  frame.epoch = 7;
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

TEST_F(RclcppFixture, PublishesFloat32FrameFromPcm16Input)
{
  rclcpp::NodeOptions options = quietGraphNodeOptions();
  options.parameter_overrides({
    rclcpp::Parameter("input_topic", "/fa_sample_format_test/input"),
    rclcpp::Parameter("output_topic", "/fa_sample_format_test/output"),
    rclcpp::Parameter("input_stream_id", "audio/test/raw"),
    rclcpp::Parameter("output.stream_id", "audio/test/float32"),
    rclcpp::Parameter("input.encoding", "PCM16LE"),
    rclcpp::Parameter("input.bit_depth", 16),
    rclcpp::Parameter("output.encoding", "FLOAT32LE"),
    rclcpp::Parameter("output.bit_depth", 32),
    rclcpp::Parameter("expected.sample_rate", 16000),
    rclcpp::Parameter("expected.channels", 1),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", false),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
    rclcpp::Parameter("diagnostics.qos.depth", 10),
    rclcpp::Parameter("diagnostics.qos.reliable", true),
  });

  auto sample_format_node = std::make_shared<fa_sample_format::FaSampleFormatNode>(options);
  auto test_node = std::make_shared<rclcpp::Node>("fa_sample_format_graph_test", quietGraphNodeOptions());

  rclcpp::QoS qos(10);
  qos.best_effort();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_sample_format_test/input",
    qos);
  std::optional<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_sample_format_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received = *msg;
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(sample_format_node);
  executor.add_node(test_node);

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (!received.has_value() && std::chrono::steady_clock::now() < deadline) {
    publisher->publish(makePcm16Frame(*test_node));
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(sample_format_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->source_id, "test-mic");
  EXPECT_EQ(received->stream_id, "audio/test/float32");
  EXPECT_EQ(received->encoding, "FLOAT32LE");
  EXPECT_EQ(received->sample_rate, 16000U);
  EXPECT_EQ(received->channels, 1U);
  EXPECT_EQ(received->bit_depth, 32U);
  EXPECT_EQ(received->layout, "interleaved");
  EXPECT_EQ(received->epoch, 7U);
  ASSERT_EQ(received->data.size(), 3U * sizeof(float));
  EXPECT_FLOAT_EQ(readFloat32Le(received->data, 0), -1.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(received->data, 1), 0.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(received->data, 2), 32767.0F / 32768.0F);
}

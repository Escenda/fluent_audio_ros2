#include "fa_decode/fa_decode_node.hpp"

#include <chrono>
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

fa_interfaces::msg::EncodedAudioChunk makeEncodedChunk(const rclcpp::Node & node)
{
  fa_interfaces::msg::EncodedAudioChunk chunk;
  chunk.header.stamp = node.now();
  chunk.source_id = "test-mic";
  chunk.stream_id = "/fa_decode_test/input";
  chunk.codec = "opus";
  chunk.container = "ogg";
  chunk.payload_format = "ogg_page";
  chunk.sample_rate = 16000;
  chunk.channels = 1;
  chunk.sequence = 3;
  chunk.media_time_ns = 40000000;
  chunk.duration_ns = 187500;
  chunk.epoch = 9;
  chunk.data = {
    0x00, 0x00,
    0x01, 0x00,
    0x02, 0x00,
  };
  return chunk;
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

TEST_F(RclcppFixture, PublishesAudioFrameFromEncodedInput)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("backend.name", "external_codec_decoder"),
    rclcpp::Parameter("backend.command.executable", "/bin/cat"),
    rclcpp::Parameter("backend.command.arguments", std::vector<std::string>{}),
    rclcpp::Parameter("backend.command.timeout_ms", 1000),
    rclcpp::Parameter("backend.command.max_output_bytes", 1024),
    rclcpp::Parameter("input_topic", "/fa_decode_test/input"),
    rclcpp::Parameter("output_topic", "/fa_decode_test/output"),
    rclcpp::Parameter("input.codec", "opus"),
    rclcpp::Parameter("input.container", "ogg"),
    rclcpp::Parameter("input.payload_format", "ogg_page"),
    rclcpp::Parameter("input.sample_rate", 16000),
    rclcpp::Parameter("input.channels", 1),
    rclcpp::Parameter("output.sample_rate", 16000),
    rclcpp::Parameter("output.channels", 1),
    rclcpp::Parameter("output.encoding", "PCM16LE"),
    rclcpp::Parameter("output.bit_depth", 16),
    rclcpp::Parameter("output.layout", "interleaved"),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", true),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
  });

  auto decode_node = std::make_shared<fa_decode::FaDecodeNode>(options);
  auto test_node = std::make_shared<rclcpp::Node>("fa_decode_graph_test");

  rclcpp::QoS qos(10);
  qos.reliable();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::EncodedAudioChunk>(
    "/fa_decode_test/input",
    qos);
  std::optional<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_decode_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received = *msg;
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(decode_node);
  executor.add_node(test_node);

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (!received.has_value() && std::chrono::steady_clock::now() < deadline) {
    publisher->publish(makeEncodedChunk(*test_node));
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(decode_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->source_id, "test-mic");
  EXPECT_EQ(received->stream_id, "/fa_decode_test/output");
  EXPECT_EQ(received->encoding, "PCM16LE");
  EXPECT_EQ(received->sample_rate, 16000U);
  EXPECT_EQ(received->channels, 1U);
  EXPECT_EQ(received->bit_depth, 16U);
  EXPECT_EQ(received->layout, "interleaved");
  EXPECT_EQ(received->epoch, 9U);
  EXPECT_EQ(received->data, makeEncodedChunk(*test_node).data);
}

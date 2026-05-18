#include "fa_encode/fa_encode_node.hpp"

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

fa_interfaces::msg::AudioFrame makePcm16Frame(const rclcpp::Node & node)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = node.now();
  frame.source_id = "test-mic";
  frame.stream_id = "fa_encode_test/input_stream";
  frame.encoding = "PCM16LE";
  frame.sample_rate = 16000;
  frame.channels = 1;
  frame.bit_depth = 16;
  frame.layout = "interleaved";
  frame.data = {
    0x00, 0x00,
    0x01, 0x00,
    0x02, 0x00,
  };
  frame.epoch = 9;
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

TEST_F(RclcppFixture, PublishesEncodedAudioChunkFromPcmInput)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("backend.name", "external_codec_encoder"),
    rclcpp::Parameter("backend.command.executable", "/bin/cat"),
    rclcpp::Parameter("backend.command.arguments", std::vector<std::string>{}),
    rclcpp::Parameter("backend.command.timeout_ms", 1000),
    rclcpp::Parameter("backend.command.max_output_bytes", 1024),
    rclcpp::Parameter("input_topic", "/fa_encode_test/input"),
    rclcpp::Parameter("output_topic", "/fa_encode_test/output"),
    rclcpp::Parameter("input_stream_id", "fa_encode_test/input_stream"),
    rclcpp::Parameter("output.stream_id", "fa_encode_test/output_stream"),
    rclcpp::Parameter("input.sample_rate", 16000),
    rclcpp::Parameter("input.channels", 1),
    rclcpp::Parameter("input.encoding", "PCM16LE"),
    rclcpp::Parameter("input.bit_depth", 16),
    rclcpp::Parameter("input.layout", "interleaved"),
    rclcpp::Parameter("output.codec", "opus"),
    rclcpp::Parameter("output.container", "ogg"),
    rclcpp::Parameter("output.payload_format", "ogg_page"),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", true),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
    rclcpp::Parameter("diagnostics.qos.depth", 10),
    rclcpp::Parameter("diagnostics.qos.reliable", true),
  });

  auto encode_node = std::make_shared<fa_encode::FaEncodeNode>(options);
  auto test_node = std::make_shared<rclcpp::Node>("fa_encode_graph_test");

  rclcpp::QoS qos(10);
  qos.reliable();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_encode_test/input",
    qos);
  std::optional<fa_interfaces::msg::EncodedAudioChunk> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::EncodedAudioChunk>(
    "/fa_encode_test/output",
    qos,
    [&received](const fa_interfaces::msg::EncodedAudioChunk::SharedPtr msg) {
      received = *msg;
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(encode_node);
  executor.add_node(test_node);

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (!received.has_value() && std::chrono::steady_clock::now() < deadline) {
    publisher->publish(makePcm16Frame(*test_node));
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(encode_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->source_id, "test-mic");
  EXPECT_EQ(received->stream_id, "fa_encode_test/output_stream");
  EXPECT_EQ(received->codec, "opus");
  EXPECT_EQ(received->container, "ogg");
  EXPECT_EQ(received->payload_format, "ogg_page");
  EXPECT_EQ(received->sample_rate, 16000U);
  EXPECT_EQ(received->channels, 1U);
  EXPECT_EQ(received->sequence, 0U);
  EXPECT_EQ(received->media_time_ns, 0U);
  EXPECT_EQ(received->duration_ns, 187500U);
  EXPECT_EQ(received->epoch, 9U);
  EXPECT_EQ(received->data, makePcm16Frame(*test_node).data);
}

#include "fa_declick/fa_declick_node.hpp"

#include <chrono>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
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
  bytes.reserve(samples.size() * sizeof(float));
  for (const float sample : samples) {
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(float));
    std::memcpy(bytes.data() + offset, &sample, sizeof(float));
  }
  return bytes;
}

std::vector<float> decodeFloat32Le(const std::vector<uint8_t> & bytes)
{
  std::vector<float> samples;
  samples.reserve(bytes.size() / sizeof(float));
  for (size_t offset = 0; offset < bytes.size(); offset += sizeof(float)) {
    float sample = 0.0F;
    std::memcpy(&sample, bytes.data() + offset, sizeof(float));
    samples.push_back(sample);
  }
  return samples;
}

fa_interfaces::msg::AudioFrame makeFloat32Frame(const std::vector<float> & samples)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp.sec = 10;
  frame.header.stamp.nanosec = 0U;
  frame.source_id = "mic-a";
  frame.stream_id = "fa_declick_test_input_stream";
  frame.encoding = "FLOAT32LE";
  frame.sample_rate = 16000;
  frame.channels = 1;
  frame.bit_depth = 32;
  frame.layout = "interleaved";
  frame.data = float32LeBytes(samples);
  frame.epoch = 7U;
  return frame;
}

rclcpp::NodeOptions validNodeOptions(
  const std::string & input_topic = "/fa_declick_test/input",
  const std::string & output_topic = "/fa_declick_test/output",
  const std::string & input_stream_id = "fa_declick_test_input_stream",
  const std::string & output_stream_id = "fa_declick_test_output_stream")
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("input_topic", input_topic),
    rclcpp::Parameter("output_topic", output_topic),
    rclcpp::Parameter("input_stream_id", input_stream_id),
    rclcpp::Parameter("output.stream_id", output_stream_id),
    rclcpp::Parameter("threshold.delta", 0.25),
    rclcpp::Parameter("window.max_samples", 1),
    rclcpp::Parameter("expected.sample_rate", 16000),
    rclcpp::Parameter("expected.channels", 1),
    rclcpp::Parameter("expected.encoding", "FLOAT32LE"),
    rclcpp::Parameter("expected.bit_depth", 32),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", true),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
    rclcpp::Parameter("diagnostics.qos.depth", 10),
    rclcpp::Parameter("diagnostics.qos.reliable", true),
  });
  return options;
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

TEST_F(RclcppFixture, PublishesDeclickedFloat32Frame)
{
  auto declick_node = std::make_shared<fa_declick::FaDeclickNode>(validNodeOptions());
  auto test_node = std::make_shared<rclcpp::Node>("fa_declick_graph_test");

  rclcpp::QoS qos(10);
  qos.reliable();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_declick_test/input",
    qos);
  std::optional<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_declick_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received = *msg;
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(declick_node);
  executor.add_node(test_node);

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (!received.has_value() && std::chrono::steady_clock::now() < deadline) {
    publisher->publish(makeFloat32Frame({0.1F, 0.9F, 0.1F}));
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(declick_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->source_id, "mic-a");
  EXPECT_EQ(received->stream_id, "fa_declick_test_output_stream");
  EXPECT_EQ(received->encoding, "FLOAT32LE");
  EXPECT_EQ(received->sample_rate, 16000U);
  EXPECT_EQ(received->channels, 1U);
  EXPECT_EQ(received->bit_depth, 32U);
  EXPECT_EQ(received->layout, "interleaved");
  EXPECT_EQ(received->epoch, 7U);
  EXPECT_EQ(decodeFloat32Le(received->data), (std::vector<float>{0.1F, 0.1F, 0.1F}));
}

TEST_F(RclcppFixture, DropsOutOfRangeSamples)
{
  auto declick_node = std::make_shared<fa_declick::FaDeclickNode>(validNodeOptions());
  auto test_node = std::make_shared<rclcpp::Node>("fa_declick_drop_test");

  rclcpp::QoS qos(10);
  qos.reliable();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_declick_test/input",
    qos);
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_declick_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(declick_node);
  executor.add_node(test_node);

  publisher->publish(makeFloat32Frame({1.25F, 0.0F, 0.0F}));
  const auto deadline = std::chrono::steady_clock::now() + 400ms;
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(declick_node);
  subscriber.reset();
  publisher.reset();

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppFixture, RejectsSameResolvedInputAndOutputTopicAtStartup)
{
  EXPECT_THROW(
    fa_declick::FaDeclickNode(validNodeOptions("fa_declick_test/same", "/fa_declick_test/same")),
    std::runtime_error);
}

TEST_F(RclcppFixture, RejectsInputStreamIdentityCollidingWithTopicAtStartup)
{
  EXPECT_THROW(
    fa_declick::FaDeclickNode(
      validNodeOptions(
        "/fa_declick_test/input", "/fa_declick_test/output", "/fa_declick_test/input",
        "fa_declick_test_output_stream")),
    std::runtime_error);
}

TEST_F(RclcppFixture, RejectsInputAndOutputStreamIdentityCollisionAtStartup)
{
  EXPECT_THROW(
    fa_declick::FaDeclickNode(
      validNodeOptions(
        "/fa_declick_test/input", "/fa_declick_test/output", "same_stream", "same_stream")),
    std::runtime_error);
}

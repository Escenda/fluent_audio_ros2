#include "fa_dc_offset_removal/fa_dc_offset_removal_node.hpp"

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
constexpr const char * kInputStreamId = "fa_dc_offset_test/input_stream";
constexpr const char * kOutputStreamId = "fa_dc_offset_test/output_stream";

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

fa_interfaces::msg::AudioFrame makeFloat32Frame(const rclcpp::Node & node)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = node.now();
  frame.source_id = "test-mic";
  frame.stream_id = kInputStreamId;
  frame.encoding = "FLOAT32LE";
  frame.sample_rate = 16000;
  frame.channels = 2;
  frame.bit_depth = 32;
  frame.layout = "interleaved";
  frame.data = float32LeBytes({1.0F, 3.0F, 3.0F, 7.0F});
  frame.epoch = 11;
  return frame;
}

rclcpp::NodeOptions validNodeOptions(
  const std::string & input_topic = "/fa_dc_offset_test/input",
  const std::string & output_topic = "/fa_dc_offset_test/output",
  const int qos_depth = 10)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("input_topic", input_topic),
    rclcpp::Parameter("output_topic", output_topic),
    rclcpp::Parameter("input_stream_id", kInputStreamId),
    rclcpp::Parameter("output.stream_id", kOutputStreamId),
    rclcpp::Parameter("expected.sample_rate", 16000),
    rclcpp::Parameter("expected.channels", 2),
    rclcpp::Parameter("expected.encoding", "FLOAT32LE"),
    rclcpp::Parameter("expected.bit_depth", 32),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("qos.depth", qos_depth),
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

TEST_F(RclcppFixture, PublishesDcOffsetRemovedFloat32Frame)
{
  auto dc_node = std::make_shared<fa_dc_offset_removal::FaDcOffsetRemovalNode>(
    validNodeOptions());
  auto test_node = std::make_shared<rclcpp::Node>("fa_dc_offset_graph_test");

  rclcpp::QoS qos(10);
  qos.reliable();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_dc_offset_test/input",
    qos);
  std::optional<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_dc_offset_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received = *msg;
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(dc_node);
  executor.add_node(test_node);

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (!received.has_value() && std::chrono::steady_clock::now() < deadline) {
    publisher->publish(makeFloat32Frame(*test_node));
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(dc_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->source_id, "test-mic");
  EXPECT_EQ(received->stream_id, kOutputStreamId);
  EXPECT_EQ(received->encoding, "FLOAT32LE");
  EXPECT_EQ(received->sample_rate, 16000U);
  EXPECT_EQ(received->channels, 2U);
  EXPECT_EQ(received->bit_depth, 32U);
  EXPECT_EQ(received->layout, "interleaved");
  EXPECT_EQ(received->epoch, 11U);
  ASSERT_EQ(received->data.size(), 4U * sizeof(float));
  EXPECT_FLOAT_EQ(readFloat32Le(received->data, 0), -1.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(received->data, 1), -2.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(received->data, 2), 1.0F);
  EXPECT_FLOAT_EQ(readFloat32Le(received->data, 3), 2.0F);
}

TEST_F(RclcppFixture, RejectsSameInputAndOutputTopicAtStartup)
{
  EXPECT_THROW(
    fa_dc_offset_removal::FaDcOffsetRemovalNode(
      validNodeOptions("/fa_dc_offset_test/same", "/fa_dc_offset_test/same")),
    std::runtime_error);
}

TEST_F(RclcppFixture, RejectsEquivalentResolvedInputAndOutputTopicAtStartup)
{
  EXPECT_THROW(
    fa_dc_offset_removal::FaDcOffsetRemovalNode(
      validNodeOptions("fa_dc_offset_test/expanded", "/fa_dc_offset_test/expanded")),
    std::runtime_error);
}

TEST_F(RclcppFixture, RejectsInvalidQosDepthAtStartup)
{
  EXPECT_THROW(
    fa_dc_offset_removal::FaDcOffsetRemovalNode(
      validNodeOptions("/fa_dc_offset_test/input", "/fa_dc_offset_test/output", 0)),
    std::runtime_error);
}

TEST_F(RclcppFixture, RejectsInputStreamThatMatchesTopicAtStartup)
{
  auto options = validNodeOptions();
  options.parameter_overrides({
    rclcpp::Parameter("input_topic", "/fa_dc_offset_test/input"),
    rclcpp::Parameter("output_topic", "/fa_dc_offset_test/output"),
    rclcpp::Parameter("input_stream_id", "fa_dc_offset_test/input"),
    rclcpp::Parameter("output.stream_id", kOutputStreamId),
    rclcpp::Parameter("expected.sample_rate", 16000),
    rclcpp::Parameter("expected.channels", 2),
    rclcpp::Parameter("expected.encoding", "FLOAT32LE"),
    rclcpp::Parameter("expected.bit_depth", 32),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", true),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
    rclcpp::Parameter("diagnostics.qos.depth", 10),
    rclcpp::Parameter("diagnostics.qos.reliable", true),
  });

  EXPECT_THROW(
    fa_dc_offset_removal::FaDcOffsetRemovalNode{options},
    std::runtime_error);
}

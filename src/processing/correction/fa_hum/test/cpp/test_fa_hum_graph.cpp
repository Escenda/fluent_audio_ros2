#include "fa_hum/fa_hum_node.hpp"

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

fa_interfaces::msg::AudioFrame makeFloat32Frame(
  const std::string & source_id,
  const uint32_t epoch,
  const int32_t stamp_sec)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp.sec = stamp_sec;
  frame.header.stamp.nanosec = 0U;
  frame.source_id = source_id;
  frame.stream_id = "fa_hum_test/input_stream";
  frame.encoding = "FLOAT32LE";
  frame.sample_rate = 16000;
  frame.channels = 1;
  frame.bit_depth = 32;
  frame.layout = "interleaved";
  frame.data = float32LeBytes({0.1F, -0.1F, 0.2F, -0.2F});
  frame.epoch = epoch;
  return frame;
}

rclcpp::NodeOptions validNodeOptions(
  const std::string & input_topic = "/fa_hum_test/input",
  const std::string & output_topic = "/fa_hum_test/output",
  const std::string & input_stream_id = "fa_hum_test/input_stream",
  const std::string & output_stream_id = "fa_hum_test/output_stream")
{
  rclcpp::NodeOptions options = quietGraphNodeOptions();
  options.parameter_overrides({
    rclcpp::Parameter("input_topic", input_topic),
    rclcpp::Parameter("output_topic", output_topic),
    rclcpp::Parameter("input_stream_id", input_stream_id),
    rclcpp::Parameter("output.stream_id", output_stream_id),
    rclcpp::Parameter("hum.frequency_hz", 60.0),
    rclcpp::Parameter("hum.harmonics", 4),
    rclcpp::Parameter("hum.q", 30.0),
    rclcpp::Parameter("expected.sample_rate", 16000),
    rclcpp::Parameter("expected.channels", 1),
    rclcpp::Parameter("expected.encoding", "FLOAT32LE"),
    rclcpp::Parameter("expected.bit_depth", 32),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", false),
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

TEST_F(RclcppFixture, PublishesHumRemovedFloat32Frame)
{
  auto hum_node = std::make_shared<fa_hum::FaHumNode>(validNodeOptions());
  auto test_node = std::make_shared<rclcpp::Node>("fa_hum_graph_test", quietGraphNodeOptions());

  rclcpp::QoS qos(10);
  qos.best_effort();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_hum_test/input",
    qos);
  std::optional<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_hum_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received = *msg;
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(hum_node);
  executor.add_node(test_node);

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (!received.has_value() && std::chrono::steady_clock::now() < deadline) {
    publisher->publish(makeFloat32Frame("mic-a", 7U, 10));
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(hum_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->source_id, "mic-a");
  EXPECT_EQ(received->stream_id, "fa_hum_test/output_stream");
  EXPECT_EQ(received->encoding, "FLOAT32LE");
  EXPECT_EQ(received->sample_rate, 16000U);
  EXPECT_EQ(received->channels, 1U);
  EXPECT_EQ(received->bit_depth, 32U);
  EXPECT_EQ(received->layout, "interleaved");
  EXPECT_EQ(received->epoch, 7U);
  EXPECT_EQ(received->data.size(), 4U * sizeof(float));
}

TEST_F(RclcppFixture, DropsOlderStampForSameSourceAndEpoch)
{
  auto hum_node = std::make_shared<fa_hum::FaHumNode>(validNodeOptions());
  auto test_node = std::make_shared<rclcpp::Node>("fa_hum_stale_stamp_test", quietGraphNodeOptions());

  rclcpp::QoS qos(10);
  qos.best_effort();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_hum_test/input",
    qos);
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_hum_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(hum_node);
  executor.add_node(test_node);

  publisher->publish(makeFloat32Frame("mic-a", 7U, 10));
  const auto first_deadline = std::chrono::steady_clock::now() + 2s;
  while (received.empty() && std::chrono::steady_clock::now() < first_deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }
  ASSERT_EQ(received.size(), 1U);

  publisher->publish(makeFloat32Frame("mic-a", 7U, 9));
  const auto stale_deadline = std::chrono::steady_clock::now() + 400ms;
  while (std::chrono::steady_clock::now() < stale_deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(hum_node);
  subscriber.reset();
  publisher.reset();

  EXPECT_EQ(received.size(), 1U);
}

TEST_F(RclcppFixture, DropsStaleEpochForSameSource)
{
  auto hum_node = std::make_shared<fa_hum::FaHumNode>(validNodeOptions());
  auto test_node = std::make_shared<rclcpp::Node>("fa_hum_stale_epoch_test", quietGraphNodeOptions());

  rclcpp::QoS qos(10);
  qos.best_effort();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_hum_test/input",
    qos);
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_hum_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(hum_node);
  executor.add_node(test_node);

  publisher->publish(makeFloat32Frame("mic-a", 2U, 10));
  const auto first_deadline = std::chrono::steady_clock::now() + 2s;
  while (received.empty() && std::chrono::steady_clock::now() < first_deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }
  ASSERT_EQ(received.size(), 1U);

  publisher->publish(makeFloat32Frame("mic-a", 1U, 11));
  const auto stale_deadline = std::chrono::steady_clock::now() + 400ms;
  while (std::chrono::steady_clock::now() < stale_deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(hum_node);
  subscriber.reset();
  publisher.reset();

  EXPECT_EQ(received.size(), 1U);
}

TEST_F(RclcppFixture, DropsFrameWhenStreamIdentityDoesNotMatchInputStream)
{
  auto hum_node = std::make_shared<fa_hum::FaHumNode>(validNodeOptions());
  auto test_node = std::make_shared<rclcpp::Node>("fa_hum_stream_identity_test", quietGraphNodeOptions());

  rclcpp::QoS qos(10);
  qos.best_effort();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_hum_test/input",
    qos);
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_hum_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(hum_node);
  executor.add_node(test_node);

  auto frame = makeFloat32Frame("mic-a", 7U, 10);
  frame.stream_id = "fa_hum_test/wrong_stream";
  publisher->publish(frame);
  const auto deadline = std::chrono::steady_clock::now() + 400ms;
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(hum_node);
  subscriber.reset();
  publisher.reset();

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppFixture, RejectsSameRawInputAndOutputTopicAtStartup)
{
  EXPECT_THROW(
    fa_hum::FaHumNode(validNodeOptions("/fa_hum_test/same", "/fa_hum_test/same")),
    std::runtime_error);
}

TEST_F(RclcppFixture, RejectsSameResolvedInputAndOutputTopicAtStartup)
{
  EXPECT_THROW(
    fa_hum::FaHumNode(validNodeOptions("fa_hum_test/same", "/fa_hum_test/same")),
    std::runtime_error);
}

TEST_F(RclcppFixture, RejectsStreamIdentityCollisionAtStartup)
{
  EXPECT_THROW(
    fa_hum::FaHumNode(
      validNodeOptions(
        "/fa_hum_test/input",
        "/fa_hum_test/output",
        "fa_hum_test/shared_stream",
        "/fa_hum_test/shared_stream")),
    std::runtime_error);
}

TEST_F(RclcppFixture, RejectsStreamIdentityThatMatchesTopicAtStartup)
{
  EXPECT_THROW(
    fa_hum::FaHumNode(
      validNodeOptions(
        "/fa_hum_test/input",
        "/fa_hum_test/output",
        "fa_hum_test/input",
        "fa_hum_test/output_stream")),
    std::runtime_error);
}

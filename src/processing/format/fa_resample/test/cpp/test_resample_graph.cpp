#include "fa_resample/fa_resample_node.hpp"
#include "fa_resample/backends/internal_linear_resampler.hpp"

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

fa_interfaces::msg::AudioFrame makeFloat32Frame(const rclcpp::Node & node)
{
  std::vector<float> samples;
  samples.reserve(480);
  for (int i = 0; i < 480; ++i) {
    samples.push_back(static_cast<float>(i) / 480.0F);
  }

  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = node.now();
  frame.source_id = "test-mic";
  frame.stream_id = "/fa_resample_test/input";
  frame.encoding = "FLOAT32LE";
  frame.sample_rate = 48000;
  frame.channels = 1;
  frame.bit_depth = 32;
  frame.layout = "interleaved";
  frame.data = fa_resample::backends::encodeFloat32Le(samples);
  frame.epoch = 11;
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

TEST_F(RclcppFixture, PublishesResampledFloat32Frame)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("target_sample_rate", 16000),
    rclcpp::Parameter("input.encoding", "FLOAT32LE"),
    rclcpp::Parameter("input.bit_depth", 32),
    rclcpp::Parameter("input.layout", "interleaved"),
    rclcpp::Parameter("output.encoding", "FLOAT32LE"),
    rclcpp::Parameter("output.bit_depth", 32),
    rclcpp::Parameter("mic.enabled", true),
    rclcpp::Parameter("mic.input_topic", "/fa_resample_test/input"),
    rclcpp::Parameter("mic.output_topic", "/fa_resample_test/output"),
    rclcpp::Parameter("ref.enabled", false),
    rclcpp::Parameter("ref.input_topic", "/fa_resample_test/ref_in"),
    rclcpp::Parameter("ref.output_topic", "/fa_resample_test/ref_out"),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", true),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
  });

  auto resample_node = std::make_shared<fa_resample::FaResampleNode>(options);
  auto test_node = std::make_shared<rclcpp::Node>("fa_resample_graph_test");

  rclcpp::QoS qos(10);
  qos.reliable();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_resample_test/input",
    qos);
  std::optional<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_resample_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received = *msg;
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(resample_node);
  executor.add_node(test_node);

  auto wrong_stream = makeFloat32Frame(*test_node);
  wrong_stream.stream_id = "/fa_resample_test/other_input";
  for (int i = 0; i < 4; ++i) {
    publisher->publish(wrong_stream);
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_FALSE(received.has_value());

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (!received.has_value() && std::chrono::steady_clock::now() < deadline) {
    publisher->publish(makeFloat32Frame(*test_node));
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(resample_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->source_id, "test-mic");
  EXPECT_EQ(received->stream_id, "/fa_resample_test/output");
  EXPECT_EQ(received->encoding, "FLOAT32LE");
  EXPECT_EQ(received->sample_rate, 16000U);
  EXPECT_EQ(received->channels, 1U);
  EXPECT_EQ(received->bit_depth, 32U);
  EXPECT_EQ(received->layout, "interleaved");
  EXPECT_EQ(received->epoch, 11U);
  EXPECT_EQ(received->data.size(), 160U * sizeof(float));
}

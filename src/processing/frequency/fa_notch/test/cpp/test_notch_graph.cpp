#include "fa_notch/fa_notch_node.hpp"

#include <chrono>
#include <cmath>
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

constexpr double kPi = 3.14159265358979323846;
constexpr const char * kInputStreamId = "audio/test/notch_input";
constexpr const char * kOutputStreamId = "audio/test/notch_output";

struct BiquadCoefficients
{
  double b0{0.0};
  double b1{0.0};
  double b2{0.0};
  double a1{0.0};
  double a2{0.0};
};

struct BiquadState
{
  double previous_input_1{0.0};
  double previous_input_2{0.0};
  double previous_output_1{0.0};
  double previous_output_2{0.0};
};

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

BiquadCoefficients notchCoefficients(
  const double sample_rate,
  const double center_hz,
  const double q)
{
  const double omega = 2.0 * kPi * center_hz / sample_rate;
  const double alpha = std::sin(omega) / (2.0 * q);
  const double cos_omega = std::cos(omega);
  const double a0 = 1.0 + alpha;
  return BiquadCoefficients{
    1.0 / a0,
    (-2.0 * cos_omega) / a0,
    1.0 / a0,
    (-2.0 * cos_omega) / a0,
    (1.0 - alpha) / a0};
}

std::vector<float> applyBiquad(
  const std::vector<float> & samples,
  const BiquadCoefficients & coefficients,
  BiquadState & state)
{
  std::vector<float> output;
  output.reserve(samples.size());
  for (const float sample : samples) {
    const double input_sample = static_cast<double>(sample);
    const double filtered =
      coefficients.b0 * input_sample +
      coefficients.b1 * state.previous_input_1 +
      coefficients.b2 * state.previous_input_2 -
      coefficients.a1 * state.previous_output_1 -
      coefficients.a2 * state.previous_output_2;
    state.previous_input_2 = state.previous_input_1;
    state.previous_input_1 = input_sample;
    state.previous_output_2 = state.previous_output_1;
    state.previous_output_1 = filtered;
    output.push_back(static_cast<float>(filtered));
  }
  return output;
}

fa_interfaces::msg::AudioFrame makeFloat32Frame(
  const rclcpp::Node & node,
  const uint32_t epoch,
  const std::vector<float> & samples)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = node.now();
  frame.source_id = "test-mic";
  frame.stream_id = kInputStreamId;
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

void waitForGraphDiscovery(
  rclcpp::executors::SingleThreadedExecutor & executor,
  const rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr & publisher,
  const rclcpp::Node::SharedPtr & test_node)
{
  const auto discovery_deadline = std::chrono::steady_clock::now() + 3s;
  while ((publisher->get_subscription_count() == 0U ||
          test_node->count_publishers("/fa_notch_test/output") == 0U) &&
         std::chrono::steady_clock::now() < discovery_deadline)
  {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  ASSERT_GT(publisher->get_subscription_count(), 0U);
  ASSERT_GT(test_node->count_publishers("/fa_notch_test/output"), 0U);
}

void waitForReceivedCount(
  rclcpp::executors::SingleThreadedExecutor & executor,
  const std::vector<fa_interfaces::msg::AudioFrame> & received,
  const size_t expected_count)
{
  const auto receive_deadline = std::chrono::steady_clock::now() + 3s;
  while (received.size() < expected_count && std::chrono::steady_clock::now() < receive_deadline) {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }
}

rclcpp::NodeOptions notchNodeOptions()
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("input_topic", "/fa_notch_test/input"),
    rclcpp::Parameter("output_topic", "/fa_notch_test/output"),
    rclcpp::Parameter("input_stream_id", kInputStreamId),
    rclcpp::Parameter("output.stream_id", kOutputStreamId),
    rclcpp::Parameter("filter.center_hz", 100.0),
    rclcpp::Parameter("filter.q", 30.0),
    rclcpp::Parameter("expected.sample_rate", 1000),
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

}  // namespace

TEST_F(RclcppFixture, PublishesNotchFilteredFloat32Frame)
{
  auto notch_node = std::make_shared<fa_notch::FaNotchNode>(notchNodeOptions());
  auto test_node = std::make_shared<rclcpp::Node>("fa_notch_graph_test");

  rclcpp::QoS qos(10);
  qos.reliable();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_notch_test/input",
    qos);
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_notch_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(notch_node);
  executor.add_node(test_node);

  waitForGraphDiscovery(executor, publisher, test_node);

  publisher->publish(makeFloat32Frame(*test_node, 17, {0.0F, 1.0F, 1.0F}));
  waitForReceivedCount(executor, received, 1U);

  publisher->publish(makeFloat32Frame(*test_node, 18, {1.0F, 1.0F}));
  waitForReceivedCount(executor, received, 2U);

  publisher->publish(makeFloat32Frame(*test_node, 17, {1.0F, 1.0F}));
  executor.spin_some(100ms);

  executor.remove_node(test_node);
  executor.remove_node(notch_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_EQ(received.size(), 2U);
  const BiquadCoefficients coefficients = notchCoefficients(1000.0, 100.0, 30.0);
  BiquadState expected_state;
  const std::vector<float> expected_first =
    applyBiquad({0.0F, 1.0F, 1.0F}, coefficients, expected_state);
  const std::vector<float> expected_second =
    applyBiquad({1.0F, 1.0F}, coefficients, expected_state);

  EXPECT_EQ(received[0].source_id, "test-mic");
  EXPECT_EQ(received[0].stream_id, kOutputStreamId);
  EXPECT_EQ(received[0].encoding, "FLOAT32LE");
  EXPECT_EQ(received[0].sample_rate, 1000U);
  EXPECT_EQ(received[0].channels, 1U);
  EXPECT_EQ(received[0].bit_depth, 32U);
  EXPECT_EQ(received[0].layout, "interleaved");
  EXPECT_EQ(received[0].epoch, 17U);
  ASSERT_EQ(received[0].data.size(), expected_first.size() * sizeof(float));
  for (size_t i = 0; i < expected_first.size(); ++i) {
    EXPECT_NEAR(readFloat32Le(received[0].data, i), expected_first.at(i), 1.0e-6F);
  }

  EXPECT_EQ(received[1].epoch, 18U);
  ASSERT_EQ(received[1].data.size(), expected_second.size() * sizeof(float));
  for (size_t i = 0; i < expected_second.size(); ++i) {
    EXPECT_NEAR(readFloat32Le(received[1].data, i), expected_second.at(i), 1.0e-6F);
  }
}

TEST_F(RclcppFixture, ResetsFilterStateOnForwardEpochGap)
{
  auto notch_node = std::make_shared<fa_notch::FaNotchNode>(notchNodeOptions());
  auto test_node = std::make_shared<rclcpp::Node>("fa_notch_epoch_gap_test");

  rclcpp::QoS qos(10);
  qos.reliable();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_notch_test/input",
    qos);
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_notch_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(notch_node);
  executor.add_node(test_node);

  waitForGraphDiscovery(executor, publisher, test_node);

  publisher->publish(makeFloat32Frame(*test_node, 30, {0.0F, 1.0F, 1.0F}));
  waitForReceivedCount(executor, received, 1U);

  publisher->publish(makeFloat32Frame(*test_node, 34, {0.25F, 1.0F}));
  waitForReceivedCount(executor, received, 2U);

  executor.remove_node(test_node);
  executor.remove_node(notch_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_EQ(received.size(), 2U);
  const BiquadCoefficients coefficients = notchCoefficients(1000.0, 100.0, 30.0);
  BiquadState fresh_state;
  const std::vector<float> expected_second =
    applyBiquad({0.25F, 1.0F}, coefficients, fresh_state);

  EXPECT_EQ(received[1].epoch, 34U);
  ASSERT_EQ(received[1].data.size(), expected_second.size() * sizeof(float));
  for (size_t i = 0; i < expected_second.size(); ++i) {
    EXPECT_NEAR(readFloat32Le(received[1].data, i), expected_second.at(i), 1.0e-6F);
  }
}

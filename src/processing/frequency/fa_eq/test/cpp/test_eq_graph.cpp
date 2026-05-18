#include "fa_eq/fa_eq_node.hpp"

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

struct EqState
{
  float previous_low_output{0.0F};
  float previous_hp_input{0.0F};
  float previous_hp_output{0.0F};
  bool initialized{false};
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

double lowAlpha(const double sample_rate, const double low_cutoff_hz)
{
  const double sample_interval_sec = 1.0 / sample_rate;
  const double rc_sec = 1.0 / (2.0 * kPi * low_cutoff_hz);
  return sample_interval_sec / (rc_sec + sample_interval_sec);
}

double highAlpha(const double sample_rate, const double high_cutoff_hz)
{
  const double sample_interval_sec = 1.0 / sample_rate;
  const double rc_sec = 1.0 / (2.0 * kPi * high_cutoff_hz);
  return rc_sec / (rc_sec + sample_interval_sec);
}

double dbToLinear(const double gain_db)
{
  return std::pow(10.0, gain_db / 20.0);
}

std::vector<float> applyEq(
  const std::vector<float> & samples,
  const double low_alpha,
  const double high_alpha,
  const double gain_low_linear,
  const double gain_mid_linear,
  const double gain_high_linear,
  EqState & state)
{
  std::vector<float> output;
  output.reserve(samples.size());
  for (const float sample : samples) {
    float low_sample = sample;
    float high_sample = 0.0F;
    if (state.initialized) {
      low_sample = static_cast<float>(
        static_cast<double>(state.previous_low_output) +
        low_alpha * (
          static_cast<double>(sample) -
          static_cast<double>(state.previous_low_output)));
      high_sample = static_cast<float>(
        high_alpha * (
          static_cast<double>(state.previous_hp_output) +
          static_cast<double>(sample) -
          static_cast<double>(state.previous_hp_input)));
    }

    const double mid_sample =
      static_cast<double>(sample) -
      static_cast<double>(low_sample) -
      static_cast<double>(high_sample);
    const double mixed =
      (static_cast<double>(low_sample) * gain_low_linear) +
      (mid_sample * gain_mid_linear) +
      (static_cast<double>(high_sample) * gain_high_linear);

    state.previous_low_output = low_sample;
    state.previous_hp_input = sample;
    state.previous_hp_output = high_sample;
    state.initialized = true;
    output.push_back(static_cast<float>(mixed));
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
  frame.stream_id = "/fa_eq_test/input";
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
          test_node->count_publishers("/fa_eq_test/output") == 0U) &&
         std::chrono::steady_clock::now() < discovery_deadline)
  {
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  ASSERT_GT(publisher->get_subscription_count(), 0U);
  ASSERT_GT(test_node->count_publishers("/fa_eq_test/output"), 0U);
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

rclcpp::NodeOptions eqNodeOptions()
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("input_topic", "/fa_eq_test/input"),
    rclcpp::Parameter("output_topic", "/fa_eq_test/output"),
    rclcpp::Parameter("low.cutoff_hz", 100.0),
    rclcpp::Parameter("high.cutoff_hz", 200.0),
    rclcpp::Parameter("gains.low_db", 3.0),
    rclcpp::Parameter("gains.mid_db", 0.0),
    rclcpp::Parameter("gains.high_db", -3.0),
    rclcpp::Parameter("expected.sample_rate", 1000),
    rclcpp::Parameter("expected.channels", 1),
    rclcpp::Parameter("expected.encoding", "FLOAT32LE"),
    rclcpp::Parameter("expected.bit_depth", 32),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", true),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
  });
  return options;
}

}  // namespace

TEST_F(RclcppFixture, PublishesThreeBandEqFloat32Frame)
{
  auto eq_node = std::make_shared<fa_eq::FaEqNode>(eqNodeOptions());
  auto test_node = std::make_shared<rclcpp::Node>("fa_eq_graph_test");

  rclcpp::QoS qos(10);
  qos.reliable();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_eq_test/input",
    qos);
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_eq_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(eq_node);
  executor.add_node(test_node);

  waitForGraphDiscovery(executor, publisher, test_node);

  publisher->publish(makeFloat32Frame(*test_node, 17, {0.0F, 0.25F, 0.25F}));
  waitForReceivedCount(executor, received, 1U);

  publisher->publish(makeFloat32Frame(*test_node, 18, {0.25F, 0.25F}));
  waitForReceivedCount(executor, received, 2U);

  publisher->publish(makeFloat32Frame(*test_node, 17, {0.25F, 0.25F}));
  executor.spin_some(100ms);

  executor.remove_node(test_node);
  executor.remove_node(eq_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_EQ(received.size(), 2U);
  const double low_alpha = lowAlpha(1000.0, 100.0);
  const double high_alpha = highAlpha(1000.0, 200.0);
  EqState expected_state;
  const std::vector<float> expected_first =
    applyEq(
      {0.0F, 0.25F, 0.25F},
      low_alpha,
      high_alpha,
      dbToLinear(3.0),
      dbToLinear(0.0),
      dbToLinear(-3.0),
      expected_state);
  const std::vector<float> expected_second =
    applyEq(
      {0.25F, 0.25F},
      low_alpha,
      high_alpha,
      dbToLinear(3.0),
      dbToLinear(0.0),
      dbToLinear(-3.0),
      expected_state);

  EXPECT_EQ(received[0].source_id, "test-mic");
  EXPECT_EQ(received[0].stream_id, "/fa_eq_test/output");
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
  auto eq_node = std::make_shared<fa_eq::FaEqNode>(eqNodeOptions());
  auto test_node = std::make_shared<rclcpp::Node>("fa_eq_epoch_gap_test");

  rclcpp::QoS qos(10);
  qos.reliable();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_eq_test/input",
    qos);
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_eq_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(eq_node);
  executor.add_node(test_node);

  waitForGraphDiscovery(executor, publisher, test_node);

  publisher->publish(makeFloat32Frame(*test_node, 30, {0.0F, 0.25F, 0.25F}));
  waitForReceivedCount(executor, received, 1U);

  publisher->publish(makeFloat32Frame(*test_node, 34, {0.1F, 0.25F}));
  waitForReceivedCount(executor, received, 2U);

  executor.remove_node(test_node);
  executor.remove_node(eq_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_EQ(received.size(), 2U);
  const double low_alpha = lowAlpha(1000.0, 100.0);
  const double high_alpha = highAlpha(1000.0, 200.0);
  EqState fresh_state;
  const std::vector<float> expected_second =
    applyEq(
      {0.1F, 0.25F},
      low_alpha,
      high_alpha,
      dbToLinear(3.0),
      dbToLinear(0.0),
      dbToLinear(-3.0),
      fresh_state);

  EXPECT_EQ(received[1].epoch, 34U);
  ASSERT_EQ(received[1].data.size(), expected_second.size() * sizeof(float));
  for (size_t i = 0; i < expected_second.size(); ++i) {
    EXPECT_NEAR(readFloat32Le(received[1].data, i), expected_second.at(i), 1.0e-6F);
  }
}

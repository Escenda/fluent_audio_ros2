#include "fa_aec_nn/fa_aec_nn_node.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include "fa_aec_nn/backends/aec_nn_backend.hpp"

namespace
{
using namespace std::chrono_literals;

constexpr const char * kInputTopic = "audio/test/aec_nn_input";
constexpr const char * kOutputTopic = "audio/test/aec_nn_output";
constexpr const char * kInputStreamId = "audio/test/aec_nn_input_stream";
constexpr const char * kOutputStreamId = "audio/test/aec_nn_output_stream";
constexpr uint32_t kSampleRate = 16000;
constexpr uint32_t kChannels = 1;
constexpr uint32_t kBitDepth = 16;
constexpr int64_t kNanosecondsPerSecond = 1000000000LL;

std::vector<rclcpp::Parameter> validParameters()
{
  return {
    rclcpp::Parameter("enabled", true),
    rclcpp::Parameter("backend.name", "passthrough"),
    rclcpp::Parameter("input_topic", kInputTopic),
    rclcpp::Parameter("output_topic", kOutputTopic),
    rclcpp::Parameter("input_stream_id", kInputStreamId),
    rclcpp::Parameter("output.stream_id", kOutputStreamId),
    rclcpp::Parameter("expected_sample_rate", static_cast<int>(kSampleRate)),
    rclcpp::Parameter("expected_channels", static_cast<int>(kChannels)),
    rclcpp::Parameter("expected.encoding", "PCM16LE"),
    rclcpp::Parameter("expected.bit_depth", static_cast<int>(kBitDepth)),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", true),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
    rclcpp::Parameter("diagnostics.qos.depth", 10),
    rclcpp::Parameter("diagnostics.qos.reliable", true),
  };
}

void replaceParameter(
  std::vector<rclcpp::Parameter> & parameters,
  const rclcpp::Parameter & replacement)
{
  for (auto & parameter : parameters) {
    if (parameter.get_name() == replacement.get_name()) {
      parameter = replacement;
      return;
    }
  }
  throw std::logic_error("test parameter replacement target is missing: " + replacement.get_name());
}

rclcpp::NodeOptions optionsWith(std::vector<rclcpp::Parameter> parameters)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides(std::move(parameters));
  return options;
}

builtin_interfaces::msg::Time stampFromNanoseconds(const int64_t nanoseconds)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(nanoseconds / kNanosecondsPerSecond);
  stamp.nanosec = static_cast<uint32_t>(nanoseconds % kNanosecondsPerSecond);
  return stamp;
}

fa_interfaces::msg::AudioFrame frameWith(
  const std::string & stream_id,
  const std::vector<uint8_t> & data,
  const uint32_t epoch)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = stampFromNanoseconds(1000000000LL + static_cast<int64_t>(epoch));
  frame.header.frame_id = stream_id;
  frame.source_id = "test-source";
  frame.stream_id = stream_id;
  frame.encoding = "PCM16LE";
  frame.sample_rate = kSampleRate;
  frame.channels = kChannels;
  frame.bit_depth = kBitDepth;
  frame.layout = "interleaved";
  frame.data = data;
  frame.epoch = epoch;
  return frame;
}

fa_aec_nn::backends::AudioChunk audioChunkWith(
  const std::vector<uint8_t> & data,
  const std::string & encoding = "PCM16LE",
  const int bit_depth = static_cast<int>(kBitDepth))
{
  fa_aec_nn::backends::AudioChunk chunk;
  chunk.sample_rate = static_cast<int>(kSampleRate);
  chunk.channels = static_cast<int>(kChannels);
  chunk.bit_depth = bit_depth;
  chunk.encoding = encoding;
  chunk.layout = "interleaved";
  chunk.data = data.data();
  chunk.data_size = data.size();
  return chunk;
}

fa_aec_nn::backends::ProcessedAudioChunk processedChunkWith(
  const std::vector<uint8_t> & data,
  const std::string & encoding = "PCM16LE",
  const int bit_depth = static_cast<int>(kBitDepth))
{
  fa_aec_nn::backends::ProcessedAudioChunk output;
  output.sample_rate = static_cast<int>(kSampleRate);
  output.channels = static_cast<int>(kChannels);
  output.bit_depth = bit_depth;
  output.encoding = encoding;
  output.layout = "interleaved";
  output.data = data;
  return output;
}

bool spinUntil(
  rclcpp::executors::SingleThreadedExecutor & executor,
  const std::function<bool()> & predicate)
{
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    executor.spin_some(10ms);
    std::this_thread::sleep_for(10ms);
  }
  return predicate();
}

void spinFor(
  rclcpp::executors::SingleThreadedExecutor & executor,
  const std::chrono::milliseconds duration)
{
  const auto deadline = std::chrono::steady_clock::now() + duration;
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(10ms);
    std::this_thread::sleep_for(10ms);
  }
}

class RclcppContractTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if (!rclcpp::ok()) {
      int argc = 0;
      char ** argv = nullptr;
      rclcpp::init(argc, argv);
    }
  }

  void TearDown() override
  {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

TEST(AecNnBackendContractTest, AcceptsSameFormatAlignedOutput)
{
  const std::vector<uint8_t> data{0x00U, 0x00U};
  const auto input = audioChunkWith(data);
  const auto output = processedChunkWith(data);

  EXPECT_TRUE(fa_aec_nn::backends::validateProcessedAudioChunk(input, output).empty());
}

TEST(AecNnBackendContractTest, RejectsEmptyOutput)
{
  const std::vector<uint8_t> data{0x00U, 0x00U};
  const std::vector<uint8_t> empty;
  const auto input = audioChunkWith(data);
  const auto output = processedChunkWith(empty);

  EXPECT_EQ(
    fa_aec_nn::backends::validateProcessedAudioChunk(input, output),
    "backend output audio data must be non-empty and PCM frame aligned");
}

TEST(AecNnBackendContractTest, RejectsMisalignedOutput)
{
  const std::vector<uint8_t> data{0x00U, 0x00U};
  const auto input = audioChunkWith(data);
  const auto output = processedChunkWith({0x00U});

  EXPECT_EQ(
    fa_aec_nn::backends::validateProcessedAudioChunk(input, output),
    "backend output audio data must be non-empty and PCM frame aligned");
}

TEST(AecNnBackendContractTest, RejectsAlignedDurationChange)
{
  const std::vector<uint8_t> data{0x00U, 0x00U};
  const auto input = audioChunkWith(data);
  const auto output = processedChunkWith({0x00U, 0x00U, 0x01U, 0x00U});

  EXPECT_EQ(
    fa_aec_nn::backends::validateProcessedAudioChunk(input, output),
    "backend output audio data size must match input audio data size");
}

TEST(AecNnBackendContractTest, RejectsOutputFormatChange)
{
  const std::vector<uint8_t> data{0x00U, 0x00U};
  const auto input = audioChunkWith(data);
  const auto output = processedChunkWith(data, "FLOAT32LE", 32);

  EXPECT_EQ(
    fa_aec_nn::backends::validateProcessedAudioChunk(input, output),
    "backend output encoding must match input encoding");
}

TEST_F(RclcppContractTest, RejectsMissingBackendAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("backend.name", ""));

  EXPECT_THROW(
    (std::make_shared<fa_aec_nn::FaAecNnNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}

TEST_F(RclcppContractTest, RejectsUnknownBackendAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("backend.name", "unknown"));

  EXPECT_THROW(
    (std::make_shared<fa_aec_nn::FaAecNnNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}

TEST_F(RclcppContractTest, PublishesPassthroughFrameWithOutputStreamId)
{
  auto node = std::make_shared<fa_aec_nn::FaAecNnNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_aec_nn_passthrough_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto input_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic, rclcpp::QoS(10).reliable());
  auto output_sub = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(output_sub, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&input_pub, &output_sub]() {
    return input_pub->get_subscription_count() > 0 && output_sub->get_publisher_count() > 0;
  }));

  const std::vector<uint8_t> data{0x00U, 0x00U, 0x00U, 0x40U};
  input_pub->publish(frameWith(kInputStreamId, data, 1U));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() == 1U;
  }));

  EXPECT_EQ(received[0].stream_id, kOutputStreamId);
  EXPECT_EQ(received[0].source_id, "test-source");
  EXPECT_EQ(received[0].sample_rate, kSampleRate);
  EXPECT_EQ(received[0].channels, kChannels);
  EXPECT_EQ(received[0].bit_depth, kBitDepth);
  EXPECT_EQ(received[0].encoding, "PCM16LE");
  EXPECT_EQ(received[0].layout, "interleaved");
  EXPECT_EQ(received[0].epoch, 1U);
  EXPECT_EQ(received[0].data, data);
}

TEST_F(RclcppContractTest, DropsFrameWhenDisabled)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("enabled", false));
  auto node = std::make_shared<fa_aec_nn::FaAecNnNode>(optionsWith(std::move(parameters)));
  auto io_node = std::make_shared<rclcpp::Node>("fa_aec_nn_disabled_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto input_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic, rclcpp::QoS(10).reliable());
  auto output_sub = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(output_sub, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&input_pub, &output_sub]() {
    return input_pub->get_subscription_count() > 0 && output_sub->get_publisher_count() > 0;
  }));

  input_pub->publish(frameWith(kInputStreamId, {0x00U, 0x00U}, 1U));
  spinFor(executor, 250ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, DropsFrameWithMismatchedStreamId)
{
  auto node = std::make_shared<fa_aec_nn::FaAecNnNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_aec_nn_stream_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto input_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic, rclcpp::QoS(10).reliable());
  auto output_sub = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(output_sub, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&input_pub, &output_sub]() {
    return input_pub->get_subscription_count() > 0 && output_sub->get_publisher_count() > 0;
  }));

  input_pub->publish(frameWith("audio/test/wrong_aec_nn_input", {0x00U, 0x00U}, 1U));
  spinFor(executor, 250ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, DropsUnsupportedRuntimeEncoding)
{
  auto node = std::make_shared<fa_aec_nn::FaAecNnNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_aec_nn_encoding_contract_io");
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto input_pub = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic, rclcpp::QoS(10).reliable());
  auto output_sub = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic, rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(output_sub, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&input_pub, &output_sub]() {
    return input_pub->get_subscription_count() > 0 && output_sub->get_publisher_count() > 0;
  }));

  auto invalid = frameWith(kInputStreamId, {0x00U, 0x00U, 0x00U, 0x00U}, 1U);
  invalid.encoding = "PCM32LE";
  invalid.bit_depth = 32U;
  input_pub->publish(invalid);
  spinFor(executor, 250ms);

  EXPECT_TRUE(received.empty());
}

TEST_F(RclcppContractTest, RejectsDisabledChannelValidationAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("expected_channels", -1));

  EXPECT_THROW(
    (std::make_shared<fa_aec_nn::FaAecNnNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}

TEST_F(RclcppContractTest, RejectsInputStreamIdMatchingTopicAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("input_stream_id", kInputTopic));

  EXPECT_THROW(
    (std::make_shared<fa_aec_nn::FaAecNnNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}

TEST_F(RclcppContractTest, RejectsInputAndOutputStreamIdCollisionAtStartup)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("output.stream_id", kInputStreamId));

  EXPECT_THROW(
    (std::make_shared<fa_aec_nn::FaAecNnNode>(optionsWith(std::move(parameters)))),
    std::runtime_error);
}
}  // namespace

#include "fa_file_in/fa_file_in_node.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

namespace
{
using namespace std::chrono_literals;

constexpr const char * kOutputTopic = "audio/test/file_in";
constexpr const char * kSourceId = "fixture_source";
constexpr const char * kStreamId = "fixture_stream";

std::filesystem::path writeFixtureFile(
  const std::string & name,
  const std::vector<uint8_t> & bytes)
{
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  stream.close();
  return path;
}

std::vector<rclcpp::Parameter> validParameters(const std::filesystem::path & file_path)
{
  return {
    rclcpp::Parameter("backend.name", "pcm_file_reader"),
    rclcpp::Parameter("file.path", file_path.string()),
    rclcpp::Parameter("output_topic", kOutputTopic),
    rclcpp::Parameter("audio.source_id", kSourceId),
    rclcpp::Parameter("audio.stream_id", kStreamId),
    rclcpp::Parameter("expected.sample_rate", 16000),
    rclcpp::Parameter("expected.channels", 1),
    rclcpp::Parameter("expected.encoding", "PCM16LE"),
    rclcpp::Parameter("expected.bit_depth", 16),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("audio.frames_per_chunk", 2),
    rclcpp::Parameter("playback.loop", false),
    rclcpp::Parameter("playback.publish_period_ms", 10),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", true),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
    rclcpp::Parameter("diagnostics.qos.depth", 10),
    rclcpp::Parameter("diagnostics.qos.reliable", true),
  };
}

rclcpp::NodeOptions optionsWith(std::vector<rclcpp::Parameter> parameters)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides(std::move(parameters));
  return options;
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

TEST_F(RclcppContractTest, FailsClosedWhenBackendNameIsUnknown)
{
  const auto fixture = writeFixtureFile("fa_file_in_unknown_backend.pcm", {0x01, 0x02});
  auto parameters = validParameters(fixture);
  parameters[0] = rclcpp::Parameter("backend.name", "hidden_decoder");

  EXPECT_THROW(
    { auto node = std::make_shared<fa_file_in::FaFileInNode>(optionsWith(parameters)); },
    std::runtime_error);
}

TEST_F(RclcppContractTest, FailsClosedWhenFilePathIsMissing)
{
  const auto fixture = writeFixtureFile("fa_file_in_missing_path_placeholder.pcm", {0x01, 0x02});
  auto parameters = validParameters(fixture);
  parameters[1] = rclcpp::Parameter("file.path", "");

  EXPECT_THROW(
    { auto node = std::make_shared<fa_file_in::FaFileInNode>(optionsWith(parameters)); },
    std::runtime_error);
}

TEST_F(RclcppContractTest, FailsClosedWhenPayloadDoesNotMatchExpectedFrameSize)
{
  const auto fixture = writeFixtureFile("fa_file_in_invalid_frame_size.pcm", {0x01, 0x02, 0x03});

  EXPECT_THROW(
    { auto node = std::make_shared<fa_file_in::FaFileInNode>(optionsWith(validParameters(fixture))); },
    std::runtime_error);
}

TEST_F(RclcppContractTest, PublishesRawPcmChunksWithoutFormatMutation)
{
  const auto fixture = writeFixtureFile(
    "fa_file_in_valid_fixture.pcm",
    {0x10, 0x00, 0x20, 0x00, 0x30, 0x00, 0x40, 0x00});

  auto node = std::make_shared<fa_file_in::FaFileInNode>(
    optionsWith(validParameters(fixture)));
  auto io_node = std::make_shared<rclcpp::Node>("fa_file_in_contract_io");

  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto subscription = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic,
    rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(subscription, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&subscription]() {
    return subscription->get_publisher_count() > 0;
  }));
  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() >= 2;
  }));

  EXPECT_EQ(received[0].source_id, kSourceId);
  EXPECT_EQ(received[0].stream_id, kStreamId);
  EXPECT_EQ(received[0].encoding, "PCM16LE");
  EXPECT_EQ(received[0].sample_rate, 16000U);
  EXPECT_EQ(received[0].channels, 1U);
  EXPECT_EQ(received[0].bit_depth, 16U);
  EXPECT_EQ(received[0].layout, "interleaved");
  EXPECT_EQ(received[0].data, (std::vector<uint8_t>{0x10, 0x00, 0x20, 0x00}));
  EXPECT_EQ(received[1].data, (std::vector<uint8_t>{0x30, 0x00, 0x40, 0x00}));
}
}  // namespace

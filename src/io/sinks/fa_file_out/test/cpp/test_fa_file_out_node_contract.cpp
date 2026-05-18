#include "fa_file_out/fa_file_out_node.hpp"

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

constexpr const char * kInputTopic = "audio/test/file_out";

std::filesystem::path targetPath(const std::string & name)
{
  const auto path = std::filesystem::temp_directory_path() / name;
  std::error_code error;
  std::filesystem::remove(path, error);
  return path;
}

void writeExistingFile(const std::filesystem::path & path)
{
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  const std::vector<uint8_t> bytes{0x99, 0x88};
  stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::vector<uint8_t> readFile(const std::filesystem::path & path)
{
  std::ifstream stream(path, std::ios::binary);
  return std::vector<uint8_t>(
    std::istreambuf_iterator<char>(stream),
    std::istreambuf_iterator<char>());
}

std::vector<rclcpp::Parameter> validParameters(const std::filesystem::path & file_path)
{
  return {
    rclcpp::Parameter("backend.name", "pcm_file_writer"),
    rclcpp::Parameter("file.path", file_path.string()),
    rclcpp::Parameter("input_topic", kInputTopic),
    rclcpp::Parameter("expected.sample_rate", 16000),
    rclcpp::Parameter("expected.channels", 1),
    rclcpp::Parameter("expected.encoding", "PCM16LE"),
    rclcpp::Parameter("expected.bit_depth", 16),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("overwrite.enabled", false),
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

fa_interfaces::msg::AudioFrame validFrame()
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = rclcpp::Clock().now();
  frame.header.frame_id = "fixture_source";
  frame.source_id = "fixture_source";
  frame.stream_id = "fixture_stream";
  frame.encoding = "PCM16LE";
  frame.sample_rate = 16000;
  frame.channels = 1;
  frame.bit_depth = 16;
  frame.layout = "interleaved";
  frame.data = {0x10, 0x00, 0x20, 0x00};
  return frame;
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
  const auto target = targetPath("fa_file_out_unknown_backend.pcm");
  auto parameters = validParameters(target);
  parameters[0] = rclcpp::Parameter("backend.name", "hidden_encoder");

  EXPECT_THROW(
    { auto node = std::make_shared<fa_file_out::FaFileOutNode>(optionsWith(parameters)); },
    std::runtime_error);
}

TEST_F(RclcppContractTest, FailsClosedWhenFilePathIsMissing)
{
  const auto target = targetPath("fa_file_out_missing_path_placeholder.pcm");
  auto parameters = validParameters(target);
  parameters[1] = rclcpp::Parameter("file.path", "");

  EXPECT_THROW(
    { auto node = std::make_shared<fa_file_out::FaFileOutNode>(optionsWith(parameters)); },
    std::runtime_error);
}

TEST_F(RclcppContractTest, FailsClosedWhenOverwriteDisabledTargetExists)
{
  const auto target = targetPath("fa_file_out_existing_target.pcm");
  writeExistingFile(target);

  EXPECT_THROW(
    { auto node = std::make_shared<fa_file_out::FaFileOutNode>(optionsWith(validParameters(target))); },
    std::runtime_error);
}

TEST_F(RclcppContractTest, WritesRawPcmPayloadWithoutFormatMutation)
{
  const auto target = targetPath("fa_file_out_valid_target.pcm");
  auto node = std::make_shared<fa_file_out::FaFileOutNode>(
    optionsWith(validParameters(target)));
  auto io_node = std::make_shared<rclcpp::Node>("fa_file_out_contract_io");
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic,
    rclcpp::QoS(10).reliable());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  publisher->publish(validFrame());
  ASSERT_TRUE(spinUntil(executor, [&target]() {
    return std::filesystem::exists(target) && std::filesystem::file_size(target) == 4U;
  }));

  EXPECT_EQ(readFile(target), (std::vector<uint8_t>{0x10, 0x00, 0x20, 0x00}));
}

TEST_F(RclcppContractTest, FailsClosedWhenIncomingFrameFormatDoesNotMatch)
{
  const auto target = targetPath("fa_file_out_mismatch_target.pcm");
  auto node = std::make_shared<fa_file_out::FaFileOutNode>(
    optionsWith(validParameters(target)));
  auto io_node = std::make_shared<rclcpp::Node>("fa_file_out_mismatch_contract_io");
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic,
    rclcpp::QoS(10).reliable());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  auto frame = validFrame();
  frame.encoding = "FLOAT32LE";
  frame.bit_depth = 32;
  publisher->publish(frame);
  ASSERT_TRUE(spinUntil(executor, [&node]() {
    return node->hasFatalError();
  }));

  EXPECT_EQ(std::filesystem::file_size(target), 0U);
}
}  // namespace

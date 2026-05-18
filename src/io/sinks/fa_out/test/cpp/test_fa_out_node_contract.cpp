#include "fa_out/fa_out_node.hpp"

#include "fa_out/backends/alsa_playback_backend.hpp"

#include <atomic>
#include <chrono>
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

struct FakeSinkState
{
  std::atomic<size_t> open_calls{0};
  std::atomic<size_t> close_calls{0};
  std::atomic<size_t> write_calls{0};
  std::atomic<size_t> frames_written{0};
  std::atomic<bool> fail_on_write{false};
  std::atomic<bool> saw_backend_config{false};
  fa_out::backends::AlsaPlaybackConfig backend_config;
};

class FakeSinkBackend final : public fa_out::backends::SinkBackend
{
public:
  explicit FakeSinkBackend(std::shared_ptr<FakeSinkState> state)
  : state_(std::move(state))
  {
  }

  fa_out::backends::SinkOpenInfo open() override
  {
    open_ = true;
    state_->open_calls.fetch_add(1);
    return fa_out::backends::SinkOpenInfo{};
  }

  void close() override
  {
    if (open_) {
      state_->close_calls.fetch_add(1);
    }
    open_ = false;
  }

  bool isOpen() const override
  {
    return open_;
  }

  bool isRunning() const override
  {
    return false;
  }

  size_t writeFrames(const uint8_t * /*data*/, const size_t frame_count) override
  {
    if (!open_) {
      throw fa_out::backends::SinkBackendError("fake sink is closed");
    }
    if (state_->fail_on_write.load()) {
      throw fa_out::backends::SinkBackendError("fake write failure");
    }
    state_->write_calls.fetch_add(1);
    state_->frames_written.fetch_add(frame_count);
    return frame_count;
  }

private:
  std::shared_ptr<FakeSinkState> state_;
  bool open_{false};
};

std::vector<rclcpp::Parameter> validParameters()
{
  return {
    rclcpp::Parameter("backend.name", "alsa_playback"),
    rclcpp::Parameter("audio.device_id", "hw:0,0"),
    rclcpp::Parameter("audio.encoding", "PCM16LE"),
    rclcpp::Parameter("audio.sample_rate", 48000),
    rclcpp::Parameter("audio.channels", 1),
    rclcpp::Parameter("audio.bit_depth", 16),
    rclcpp::Parameter("audio.alsa.buffer_frames", 16384),
    rclcpp::Parameter("audio.alsa.period_frames", 4096),
    rclcpp::Parameter("queue.max_frames", 32),
    rclcpp::Parameter("audio.chunk_duration_ms", 30),
    rclcpp::Parameter("audio.qos.depth", 10),
    rclcpp::Parameter("audio.qos.reliable", true),
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

fa_out::FaOutNode::BackendFactory factoryFor(const std::shared_ptr<FakeSinkState> & state)
{
  return [state](const fa_out::backends::AlsaPlaybackConfig & config) {
    state->backend_config = config;
    state->saw_backend_config.store(true);
    return std::make_unique<FakeSinkBackend>(state);
  };
}

fa_interfaces::msg::AudioFrame validFrame()
{
  fa_interfaces::msg::AudioFrame msg;
  msg.header.stamp = rclcpp::Clock().now();
  msg.header.frame_id = "test-request";
  msg.source_id = "test-source";
  msg.stream_id = "audio/output/frame";
  msg.encoding = "PCM16LE";
  msg.sample_rate = 48000;
  msg.channels = 1;
  msg.bit_depth = 16;
  msg.layout = "interleaved";
  msg.data = {0, 0, 1, 0};
  msg.epoch = 1;
  return msg;
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
    if (!rclcpp::ok()) {
      return predicate();
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

TEST_F(RclcppContractTest, RejectsUnsupportedBackendBeforeOpeningSink)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("backend.name", "unknown_sink"));
  const auto state = std::make_shared<FakeSinkState>();

  EXPECT_THROW(
    {
      auto node = std::make_shared<fa_out::FaOutNode>(
        optionsWith(std::move(parameters)),
        factoryFor(state));
    },
    std::invalid_argument);
  EXPECT_EQ(state->open_calls.load(), 0u);
}

TEST_F(RclcppContractTest, RejectsAlsaPluginSinkBeforeOpeningSink)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("audio.device_id", "default"));
  const auto state = std::make_shared<FakeSinkState>();

  EXPECT_THROW(
    {
      auto node = std::make_shared<fa_out::FaOutNode>(
        optionsWith(std::move(parameters)),
        factoryFor(state));
    },
    std::invalid_argument);
  EXPECT_EQ(state->open_calls.load(), 0u);
}

TEST_F(RclcppContractTest, PassesExplicitConfigToSinkBackend)
{
  const auto state = std::make_shared<FakeSinkState>();
  auto node = std::make_shared<fa_out::FaOutNode>(
    optionsWith(validParameters()),
    factoryFor(state));

  EXPECT_TRUE(state->saw_backend_config.load());
  EXPECT_EQ(state->backend_config.device_id, "hw:0,0");
  EXPECT_EQ(state->backend_config.encoding, "PCM16LE");
  EXPECT_EQ(state->backend_config.sample_rate, 48000u);
  EXPECT_EQ(state->backend_config.channels, 1u);
  EXPECT_EQ(state->backend_config.bit_depth, 16u);
  EXPECT_EQ(state->backend_config.buffer_frames, 16384u);
  EXPECT_EQ(state->backend_config.period_frames, 4096u);
}

TEST_F(RclcppContractTest, WritesOnlyFramesMatchingTheConfiguredSinkContract)
{
  const auto state = std::make_shared<FakeSinkState>();
  auto node = std::make_shared<fa_out::FaOutNode>(
    optionsWith(validParameters()),
    factoryFor(state));
  auto publisher_node = std::make_shared<rclcpp::Node>("fa_out_contract_publisher");
  auto publisher = publisher_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "audio/output/frame", rclcpp::QoS(10).reliable());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(publisher_node);

  auto invalid = validFrame();
  invalid.encoding = "FLOAT32LE";
  for (int i = 0; i < 4; ++i) {
    publisher->publish(invalid);
    executor.spin_some(10ms);
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_EQ(state->write_calls.load(), 0u);
  EXPECT_FALSE(node->hasFatalError());

  auto wrong_stream = validFrame();
  wrong_stream.stream_id = "audio/tts/frame";
  for (int i = 0; i < 4; ++i) {
    publisher->publish(wrong_stream);
    executor.spin_some(10ms);
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_EQ(state->write_calls.load(), 0u);
  EXPECT_FALSE(node->hasFatalError());

  publisher->publish(validFrame());
  EXPECT_TRUE(spinUntil(executor, [&state]() {
    return state->frames_written.load() == 2u;
  }));
  EXPECT_EQ(state->write_calls.load(), 1u);
}

TEST_F(RclcppContractTest, WriteFailureFailsClosed)
{
  const auto state = std::make_shared<FakeSinkState>();
  state->fail_on_write.store(true);
  auto node = std::make_shared<fa_out::FaOutNode>(
    optionsWith(validParameters()),
    factoryFor(state));
  auto publisher_node = std::make_shared<rclcpp::Node>("fa_out_contract_failure_publisher");
  auto publisher = publisher_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "audio/output/frame", rclcpp::QoS(10).reliable());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(publisher_node);

  publisher->publish(validFrame());
  EXPECT_TRUE(spinUntil(executor, [&node]() {
    return node->hasFatalError();
  }));
  EXPECT_TRUE(node->hasFatalError());
}

}  // namespace

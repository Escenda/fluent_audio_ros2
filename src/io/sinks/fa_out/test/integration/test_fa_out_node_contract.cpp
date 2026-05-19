#include "fa_out/fa_out_node.hpp"

#include "fa_out/backends/alsa_playback_backend.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
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

rclcpp::NodeOptions quietContractNodeOptions()
{
  rclcpp::NodeOptions options;
  options.start_parameter_services(false);
  options.start_parameter_event_publisher(false);
  options.enable_rosout(false);
  return options;
}

struct FakeSinkState
{
  std::atomic<size_t> open_calls{0};
  std::atomic<size_t> close_calls{0};
  std::atomic<size_t> write_calls{0};
  std::atomic<size_t> frames_written{0};
  std::atomic<bool> fail_on_write{false};
  std::atomic<bool> saw_backend_config{false};
  fa_out::backends::AlsaPlaybackConfig backend_config;
  size_t bytes_per_frame{0};
  std::mutex written_bytes_mutex;
  std::vector<uint8_t> written_bytes;
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

  size_t writeFrames(const uint8_t * data, const size_t frame_count) override
  {
    if (!open_) {
      throw fa_out::backends::SinkBackendError("fake sink is closed");
    }
    if (state_->fail_on_write.load()) {
      throw fa_out::backends::SinkBackendError("fake write failure");
    }
    state_->write_calls.fetch_add(1);
    state_->frames_written.fetch_add(frame_count);
    const size_t bytes_to_write = frame_count * state_->bytes_per_frame;
    {
      std::lock_guard<std::mutex> lock(state_->written_bytes_mutex);
      state_->written_bytes.insert(state_->written_bytes.end(), data, data + bytes_to_write);
    }
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
    rclcpp::Parameter("input_topic", "fa_out_contract/input"),
    rclcpp::Parameter("input_stream_id", "audio/playback/main"),
    rclcpp::Parameter("playback_done_topic", "fa_out_contract/playback_done"),
    rclcpp::Parameter("playback_control_service", "fa_out_contract/playback_control"),
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
    rclcpp::Parameter("lifecycle.qos.depth", 10),
    rclcpp::Parameter("lifecycle.qos.reliable", true),
  };
}

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

std::vector<rclcpp::Parameter> validFileParameters(const std::filesystem::path & file_path)
{
  return {
    rclcpp::Parameter("backend.name", "pcm_file_writer"),
    rclcpp::Parameter("file.path", file_path.string()),
    rclcpp::Parameter("input_topic", "fa_out_contract/file_input"),
    rclcpp::Parameter("input_stream_id", "audio/playback/main"),
    rclcpp::Parameter("playback_done_topic", "fa_out_contract/file_playback_done"),
    rclcpp::Parameter("playback_control_service", "fa_out_contract/file_playback_control"),
    rclcpp::Parameter("audio.encoding", "PCM16LE"),
    rclcpp::Parameter("audio.sample_rate", 48000),
    rclcpp::Parameter("audio.channels", 1),
    rclcpp::Parameter("audio.bit_depth", 16),
    rclcpp::Parameter("queue.max_frames", 32),
    rclcpp::Parameter("audio.chunk_duration_ms", 1),
    rclcpp::Parameter("audio.qos.depth", 10),
    rclcpp::Parameter("audio.qos.reliable", true),
    rclcpp::Parameter("lifecycle.qos.depth", 10),
    rclcpp::Parameter("lifecycle.qos.reliable", true),
    rclcpp::Parameter("overwrite.enabled", false),
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

void removeParameter(std::vector<rclcpp::Parameter> & parameters, const std::string & name)
{
  const auto original_size = parameters.size();
  parameters.erase(
    std::remove_if(
      parameters.begin(),
      parameters.end(),
      [&name](const rclcpp::Parameter & parameter) {
        return parameter.get_name() == name;
      }),
    parameters.end());
  if (parameters.size() == original_size) {
    throw std::logic_error("test parameter removal target is missing: " + name);
  }
}

rclcpp::NodeOptions optionsWith(std::vector<rclcpp::Parameter> parameters)
{
  rclcpp::NodeOptions options = quietContractNodeOptions();
  options.parameter_overrides(std::move(parameters));
  return options;
}

fa_out::FaOutNode::BackendFactory factoryFor(const std::shared_ptr<FakeSinkState> & state)
{
  return [state](const fa_out::backends::AlsaPlaybackConfig & config) {
    state->backend_config = config;
    state->bytes_per_frame = config.channels * (config.bit_depth / 8u);
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
  msg.stream_id = "audio/playback/main";
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

TEST_F(RclcppContractTest, RejectsMissingBackendNameBeforeOpeningSink)
{
  auto parameters = validParameters();
  removeParameter(parameters, "backend.name");
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

TEST_F(RclcppContractTest, RejectsStreamIdMatchingInputTopicBeforeOpeningSink)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("input_topic", "/fa_out_contract/input"));
  replaceParameter(parameters, rclcpp::Parameter("input_stream_id", "fa_out_contract/input"));
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

TEST_F(RclcppContractTest, FileBackendFailsClosedWhenFilePathIsMissing)
{
  const auto target = targetPath("fa_out_missing_file_path_placeholder.pcm");
  auto parameters = validFileParameters(target);
  replaceParameter(parameters, rclcpp::Parameter("file.path", ""));
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

TEST_F(RclcppContractTest, FileBackendFailsClosedWhenOverwriteDisabledTargetExists)
{
  const auto target = targetPath("fa_out_existing_file_target.pcm");
  writeExistingFile(target);
  const auto state = std::make_shared<FakeSinkState>();

  EXPECT_THROW(
    {
      auto node = std::make_shared<fa_out::FaOutNode>(
        optionsWith(validFileParameters(target)),
        factoryFor(state));
    },
    std::runtime_error);
  EXPECT_EQ(state->open_calls.load(), 0u);
}

TEST_F(RclcppContractTest, FileBackendWritesRawPcmPayloadWithoutFormatMutation)
{
  const auto target = targetPath("fa_out_valid_file_target.pcm");
  const auto state = std::make_shared<FakeSinkState>();
  auto node = std::make_shared<fa_out::FaOutNode>(
    optionsWith(validFileParameters(target)),
    factoryFor(state));
  auto publisher_node = std::make_shared<rclcpp::Node>("fa_out_file_contract_publisher", quietContractNodeOptions());
  auto publisher = publisher_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "fa_out_contract/file_input",
    rclcpp::QoS(10).reliable());
  auto done_node = std::make_shared<rclcpp::Node>("fa_out_file_contract_done_watcher", quietContractNodeOptions());
  std::mutex done_mutex;
  std::vector<fa_interfaces::msg::PlaybackDone> done_messages;
  auto done_sub = done_node->create_subscription<fa_interfaces::msg::PlaybackDone>(
    "fa_out_contract/file_playback_done", rclcpp::QoS(10).reliable(),
    [&done_mutex, &done_messages](const fa_interfaces::msg::PlaybackDone::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(done_mutex);
      done_messages.push_back(*msg);
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(publisher_node);
  executor.add_node(done_node);
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  const auto frame = validFrame();
  publisher->publish(frame);
  ASSERT_TRUE(spinUntil(executor, [&target]() {
    return std::filesystem::exists(target) && std::filesystem::file_size(target) == 4U;
  }));

  EXPECT_EQ(readFile(target), frame.data);
  EXPECT_EQ(state->open_calls.load(), 0u);
  EXPECT_TRUE(spinUntil(executor, [&done_mutex, &done_messages]() {
    std::lock_guard<std::mutex> lock(done_mutex);
    return done_messages.size() == 1u;
  }));
}

TEST_F(RclcppContractTest, WritesOnlyFramesMatchingTheConfiguredSinkContract)
{
  const auto state = std::make_shared<FakeSinkState>();
  auto node = std::make_shared<fa_out::FaOutNode>(
    optionsWith(validParameters()),
    factoryFor(state));
  auto publisher_node = std::make_shared<rclcpp::Node>("fa_out_contract_publisher", quietContractNodeOptions());
  auto publisher = publisher_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "fa_out_contract/input", rclcpp::QoS(10).reliable());
  auto done_node = std::make_shared<rclcpp::Node>("fa_out_contract_done_watcher", quietContractNodeOptions());
  std::mutex done_mutex;
  std::vector<fa_interfaces::msg::PlaybackDone> done_messages;
  auto done_sub = done_node->create_subscription<fa_interfaces::msg::PlaybackDone>(
    "fa_out_contract/playback_done", rclcpp::QoS(10).reliable(),
    [&done_mutex, &done_messages](const fa_interfaces::msg::PlaybackDone::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(done_mutex);
      done_messages.push_back(*msg);
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(publisher_node);
  executor.add_node(done_node);

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

  const auto valid = validFrame();
  publisher->publish(valid);
  EXPECT_TRUE(spinUntil(executor, [&state]() {
    return state->frames_written.load() == 2u;
  }));
  EXPECT_EQ(state->write_calls.load(), 1u);
  {
    std::lock_guard<std::mutex> lock(state->written_bytes_mutex);
    EXPECT_EQ(state->written_bytes, valid.data);
  }
  EXPECT_TRUE(spinUntil(executor, [&done_mutex, &done_messages]() {
    std::lock_guard<std::mutex> lock(done_mutex);
    return done_messages.size() == 1u;
  }));
  {
    std::lock_guard<std::mutex> lock(done_mutex);
    ASSERT_EQ(done_messages.size(), 1u);
    EXPECT_EQ(done_messages.front().request_id, "test-request");
    EXPECT_EQ(done_messages.front().epoch, 1u);
  }
}

TEST_F(RclcppContractTest, WriteFailureFailsClosed)
{
  const auto state = std::make_shared<FakeSinkState>();
  state->fail_on_write.store(true);
  auto node = std::make_shared<fa_out::FaOutNode>(
    optionsWith(validParameters()),
    factoryFor(state));
  auto publisher_node = std::make_shared<rclcpp::Node>("fa_out_contract_failure_publisher", quietContractNodeOptions());
  auto publisher = publisher_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "fa_out_contract/input", rclcpp::QoS(10).reliable());

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

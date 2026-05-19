#include "fa_in/fa_in_node.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
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

constexpr const char * kFileOutputTopic = "audio/test/file_in";
constexpr const char * kFileSourceId = "fixture_source";
constexpr const char * kFileStreamId = "fixture_stream";

struct FakeSourceState
{
  std::atomic<size_t> list_calls{0};
  std::atomic<size_t> select_calls{0};
  std::atomic<size_t> open_calls{0};
  std::atomic<size_t> close_calls{0};
  std::atomic<size_t> drop_calls{0};
  std::atomic<size_t> read_calls{0};
  std::atomic<bool> fail_on_read{false};
  std::atomic<bool> fail_list_devices{false};
  std::atomic<bool> fail_open_on_switch_target{false};
  std::vector<fa_in::backends::DeviceInfo> devices{
    fa_in::backends::DeviceInfo{"hw:0,0", "Primary Mic", 2, 44100},
    fa_in::backends::DeviceInfo{"hw:1,0", "Backup Mic", 4, 96000},
  };
  std::vector<uint8_t> payload = []() {
    std::vector<uint8_t> bytes(96);
    for (size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<uint8_t>((i * 7u + 3u) % 251u);
    }
    return bytes;
  }();
  std::mutex opened_device_mutex;
  std::string opened_device;
};

std::string displayName(const fa_in::backends::DeviceInfo & device)
{
  return device.name;
}

std::string openedDevice(const std::shared_ptr<FakeSourceState> & state)
{
  std::lock_guard<std::mutex> lock(state->opened_device_mutex);
  return state->opened_device;
}

class FakeSourceBackend final : public fa_in::backends::SourceBackend
{
public:
  explicit FakeSourceBackend(std::shared_ptr<FakeSourceState> state)
  : state_(std::move(state))
  {
  }

  std::vector<fa_in::backends::DeviceInfo> listDevices() const override
  {
    state_->list_calls.fetch_add(1);
    if (state_->fail_list_devices.load()) {
      throw fa_in::backends::BackendError("fake source enumeration failure");
    }
    return state_->devices;
  }

  fa_in::backends::DeviceInfo selectDevice(
    const fa_in::backends::DeviceSelector & selector) const override
  {
    state_->select_calls.fetch_add(1);
    const auto devices = listDevices();
    if (selector.mode == "index") {
      if (selector.index >= 0 && static_cast<size_t>(selector.index) < devices.size()) {
        return devices[selector.index];
      }
      throw fa_in::backends::BackendError("fake source index was not found");
    }
    if (selector.mode == "id") {
      for (const auto & device : devices) {
        if (device.id == selector.identifier) {
          return device;
        }
      }
      throw fa_in::backends::BackendError("fake source id was not found");
    }
    if (selector.mode == "name") {
      for (const auto & device : devices) {
        if (displayName(device) == selector.identifier) {
          return device;
        }
      }
      throw fa_in::backends::BackendError("fake source name was not found");
    }
    throw fa_in::backends::BackendError("fake source selector mode is unsupported");
  }

  size_t open(
    const std::string & device_id,
    const fa_in::backends::AudioFormat & format,
    const size_t requested_frames) override
  {
    state_->open_calls.fetch_add(1);
    if (format.sample_rate != 48000 || format.channels != 1 || format.bit_depth != 16 ||
      format.encoding != "PCM16LE" || format.layout != "interleaved")
    {
      throw fa_in::backends::BackendError("fake source received unexpected format contract");
    }
    if (state_->fail_open_on_switch_target.load() && device_id == "hw:1,0") {
      throw fa_in::backends::BackendError("fake source reopen failure");
    }
    {
      std::lock_guard<std::mutex> lock(state_->opened_device_mutex);
      state_->opened_device = device_id;
    }
    bytes_per_frame_ =
      static_cast<size_t>(format.channels) * static_cast<size_t>(format.bit_depth / 8u);
    open_.store(true);
    return requested_frames;
  }

  void close() override
  {
    if (open_.exchange(false)) {
      state_->close_calls.fetch_add(1);
    }
  }

  void drop() override
  {
    state_->drop_calls.fetch_add(1);
  }

  fa_in::backends::ReadResult read(uint8_t * data, const size_t frames) override
  {
    state_->read_calls.fetch_add(1);
    if (!open_.load()) {
      return fa_in::backends::ReadResult{
        fa_in::backends::ReadStatus::kError,
        0,
        "fake source is closed"};
    }
    if (state_->fail_on_read.load()) {
      return fa_in::backends::ReadResult{
        fa_in::backends::ReadStatus::kError,
        0,
        "fake read failure"};
    }

    const size_t bytes_to_copy = frames * bytes_per_frame_;
    if (bytes_to_copy != state_->payload.size()) {
      return fa_in::backends::ReadResult{
        fa_in::backends::ReadStatus::kError,
        0,
        "fake source payload size does not match requested frame count"};
    }
    std::copy(state_->payload.begin(), state_->payload.end(), data);
    std::this_thread::sleep_for(2ms);
    return fa_in::backends::ReadResult{fa_in::backends::ReadStatus::kOk, frames, ""};
  }

private:
  std::shared_ptr<FakeSourceState> state_;
  std::atomic<bool> open_{false};
  size_t bytes_per_frame_{2};
};

std::vector<rclcpp::Parameter> validParameters()
{
  return {
    rclcpp::Parameter("backend.name", "alsa_capture"),
    rclcpp::Parameter("output_topic", "fa_in_contract/output"),
    rclcpp::Parameter("audio.device_selector.mode", "id"),
    rclcpp::Parameter("audio.device_selector.identifier", "hw:0,0"),
    rclcpp::Parameter("audio.device_selector.index", -1),
    rclcpp::Parameter("audio.sample_rate", 48000),
    rclcpp::Parameter("audio.channels", 1),
    rclcpp::Parameter("audio.bit_depth", 16),
    rclcpp::Parameter("audio.chunk_ms", 1),
    rclcpp::Parameter("audio.encoding", "PCM16LE"),
    rclcpp::Parameter("audio.stream_id", "audio/raw/mic"),
    rclcpp::Parameter("audio.layout", "interleaved"),
    rclcpp::Parameter("audio.qos.depth", 10),
    rclcpp::Parameter("audio.qos.reliable", false),
    rclcpp::Parameter("diagnostics.qos.depth", 10),
    rclcpp::Parameter("diagnostics.qos.reliable", false),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
  };
}

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

std::vector<rclcpp::Parameter> validFileParameters(const std::filesystem::path & file_path)
{
  return {
    rclcpp::Parameter("backend.name", "pcm_file_reader"),
    rclcpp::Parameter("file.path", file_path.string()),
    rclcpp::Parameter("output_topic", kFileOutputTopic),
    rclcpp::Parameter("audio.source_id", kFileSourceId),
    rclcpp::Parameter("audio.sample_rate", 1000),
    rclcpp::Parameter("audio.channels", 1),
    rclcpp::Parameter("audio.bit_depth", 16),
    rclcpp::Parameter("audio.chunk_ms", 2),
    rclcpp::Parameter("audio.encoding", "PCM16LE"),
    rclcpp::Parameter("audio.stream_id", kFileStreamId),
    rclcpp::Parameter("audio.layout", "interleaved"),
    rclcpp::Parameter("playback.loop", true),
    rclcpp::Parameter("audio.qos.depth", 10),
    rclcpp::Parameter("audio.qos.reliable", false),
    rclcpp::Parameter("diagnostics.qos.depth", 10),
    rclcpp::Parameter("diagnostics.qos.reliable", false),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
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
  rclcpp::NodeOptions options = quietContractNodeOptions();
  options.parameter_overrides(std::move(parameters));
  return options;
}

fa_in::FaInNode::BackendFactory factoryFor(const std::shared_ptr<FakeSourceState> & state)
{
  return [state]() {
    return std::make_unique<FakeSourceBackend>(state);
  };
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
    try {
      executor.spin_some(10ms);
    } catch (const rclcpp::exceptions::RCLError &) {
      return predicate();
    }
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

TEST_F(RclcppContractTest, RejectsUnsupportedBackendBeforeCreatingBackend)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("backend.name", "unknown_source"));
  std::atomic<size_t> factory_calls{0};

  EXPECT_THROW(
    {
      auto node = std::make_shared<fa_in::FaInNode>(
        optionsWith(std::move(parameters)),
        [&factory_calls]() {
          factory_calls.fetch_add(1);
          return std::unique_ptr<fa_in::backends::SourceBackend>{};
        });
    },
    std::runtime_error);
  EXPECT_EQ(factory_calls.load(), 0u);
}

TEST_F(RclcppContractTest, RejectsStreamIdMatchingOutputTopicBeforeCreatingBackend)
{
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("output_topic", "/fa_in_contract/output"));
  replaceParameter(parameters, rclcpp::Parameter("audio.stream_id", "fa_in_contract/output"));
  std::atomic<size_t> factory_calls{0};

  EXPECT_THROW(
    {
      auto node = std::make_shared<fa_in::FaInNode>(
        optionsWith(std::move(parameters)),
        [&factory_calls]() {
          factory_calls.fetch_add(1);
          return std::unique_ptr<fa_in::backends::SourceBackend>{};
        });
    },
    std::runtime_error);
  EXPECT_EQ(factory_calls.load(), 0u);
}

TEST_F(RclcppContractTest, PublishesConfiguredMetadataAndRawSourcePayload)
{
  const auto state = std::make_shared<FakeSourceState>();
  auto node = std::make_shared<fa_in::FaInNode>(
    optionsWith(validParameters()),
    factoryFor(state));
  auto subscriber_node = std::make_shared<rclcpp::Node>("fa_in_contract_subscriber", quietContractNodeOptions());
  std::optional<fa_interfaces::msg::AudioFrame> received;
  auto subscription = subscriber_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "fa_in_contract/output",
    rclcpp::QoS(10).best_effort(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received = *msg;
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(subscriber_node);

  EXPECT_TRUE(spinUntil(executor, [&received]() {
    return received.has_value();
  }));
  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->source_id, "hw:0,0");
  EXPECT_EQ(received->stream_id, "audio/raw/mic");
  EXPECT_EQ(received->encoding, "PCM16LE");
  EXPECT_EQ(received->sample_rate, 48000u);
  EXPECT_EQ(received->channels, 1u);
  EXPECT_EQ(received->bit_depth, 16u);
  EXPECT_EQ(received->layout, "interleaved");
  EXPECT_EQ(received->data, state->payload);
  EXPECT_GE(state->read_calls.load(), 1u);
  EXPECT_FALSE(node->hasFatalError());
}

TEST_F(RclcppContractTest, FileBackendPublishesRawPcmChunksWithoutFormatMutation)
{
  const auto fixture = writeFixtureFile(
    "fa_in_file_backend_valid_fixture.pcm",
    {0x10, 0x00, 0x20, 0x00, 0x30, 0x00, 0x40, 0x00});

  auto subscriber_node = std::make_shared<rclcpp::Node>("fa_in_file_backend_subscriber", quietContractNodeOptions());
  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto subscription = subscriber_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kFileOutputTopic,
    rclcpp::QoS(10).best_effort(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  auto node = std::make_shared<fa_in::FaInNode>(optionsWith(validFileParameters(fixture)));

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(subscriber_node);

  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return received.size() >= 4;
  }));

  const std::vector<uint8_t> first_chunk{0x10, 0x00, 0x20, 0x00};
  const std::vector<uint8_t> second_chunk{0x30, 0x00, 0x40, 0x00};
  bool saw_first_chunk = false;
  bool saw_second_chunk = false;
  for (const auto & frame : received) {
    EXPECT_EQ(frame.source_id, kFileSourceId);
    EXPECT_EQ(frame.stream_id, kFileStreamId);
    EXPECT_EQ(frame.encoding, "PCM16LE");
    EXPECT_EQ(frame.sample_rate, 1000u);
    EXPECT_EQ(frame.channels, 1u);
    EXPECT_EQ(frame.bit_depth, 16u);
    EXPECT_EQ(frame.layout, "interleaved");
    saw_first_chunk = saw_first_chunk || frame.data == first_chunk;
    saw_second_chunk = saw_second_chunk || frame.data == second_chunk;
  }
  EXPECT_TRUE(saw_first_chunk);
  EXPECT_TRUE(saw_second_chunk);
  EXPECT_FALSE(node->hasFatalError());
}

TEST_F(RclcppContractTest, FileBackendFailsClosedWhenFilePathIsMissing)
{
  const auto fixture = writeFixtureFile("fa_in_file_backend_missing_path.pcm", {0x01, 0x02});
  auto parameters = validFileParameters(fixture);
  replaceParameter(parameters, rclcpp::Parameter("file.path", ""));

  EXPECT_THROW(
    { auto node = std::make_shared<fa_in::FaInNode>(optionsWith(parameters)); },
    std::runtime_error);
}

TEST_F(RclcppContractTest, NameSelectorMatchesDisplayNameOnly)
{
  const auto state = std::make_shared<FakeSourceState>();
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("audio.device_selector.mode", "name"));
  replaceParameter(parameters, rclcpp::Parameter("audio.device_selector.identifier", "Primary Mic"));

  auto node = std::make_shared<fa_in::FaInNode>(
    optionsWith(std::move(parameters)),
    factoryFor(state));

  EXPECT_EQ(openedDevice(state), "hw:0,0");
  EXPECT_FALSE(node->hasFatalError());
}

TEST_F(RclcppContractTest, NameSelectorRejectsRawId)
{
  const auto state = std::make_shared<FakeSourceState>();
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("audio.device_selector.mode", "name"));
  replaceParameter(parameters, rclcpp::Parameter("audio.device_selector.identifier", "hw:0,0"));

  EXPECT_THROW(
    {
      auto node = std::make_shared<fa_in::FaInNode>(
        optionsWith(std::move(parameters)),
        factoryFor(state));
    },
    std::runtime_error);
  EXPECT_TRUE(openedDevice(state).empty());
}

TEST_F(RclcppContractTest, NameSelectorDoesNotFallbackToRawIdWhenDisplayNameIsEmpty)
{
  const auto state = std::make_shared<FakeSourceState>();
  state->devices = {
    fa_in::backends::DeviceInfo{"hw:2,0", "", 2, 48000},
  };
  auto parameters = validParameters();
  replaceParameter(parameters, rclcpp::Parameter("audio.device_selector.mode", "name"));
  replaceParameter(parameters, rclcpp::Parameter("audio.device_selector.identifier", "hw:2,0"));

  EXPECT_THROW(
    {
      auto node = std::make_shared<fa_in::FaInNode>(
        optionsWith(std::move(parameters)),
        factoryFor(state));
    },
    std::runtime_error);
  EXPECT_TRUE(openedDevice(state).empty());
}

TEST_F(RclcppContractTest, ListDevicesSurfacesBackendEnumerationFailure)
{
  const auto state = std::make_shared<FakeSourceState>();
  auto node = std::make_shared<fa_in::FaInNode>(
    optionsWith(validParameters()),
    factoryFor(state));
  auto client_node = std::make_shared<rclcpp::Node>("fa_in_contract_list_devices_client", quietContractNodeOptions());
  auto client = client_node->create_client<fa_interfaces::srv::ListDevices>("list_devices");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(client_node);

  ASSERT_TRUE(spinUntil(executor, [&client]() {
    return client->service_is_ready();
  }));
  state->fail_list_devices.store(true);
  auto request = std::make_shared<fa_interfaces::srv::ListDevices::Request>();
  request->refresh = true;
  auto future = client->async_send_request(request);

  ASSERT_TRUE(spinUntil(executor, [&future]() {
    return future.wait_for(0ms) == std::future_status::ready;
  }));
  const auto response = future.get();
  EXPECT_FALSE(response->success);
  EXPECT_EQ(response->message, "fake source enumeration failure");
  EXPECT_EQ(response->active_device_id, "hw:0,0");
  EXPECT_FALSE(node->hasFatalError());
}

TEST_F(RclcppContractTest, ListDevicesReturnsBackendReportedCapabilities)
{
  const auto state = std::make_shared<FakeSourceState>();
  auto node = std::make_shared<fa_in::FaInNode>(
    optionsWith(validParameters()),
    factoryFor(state));
  auto client_node = std::make_shared<rclcpp::Node>("fa_in_contract_list_capabilities_client", quietContractNodeOptions());
  auto client = client_node->create_client<fa_interfaces::srv::ListDevices>("list_devices");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(client_node);

  ASSERT_TRUE(spinUntil(executor, [&client]() {
    return client->service_is_ready();
  }));
  auto request = std::make_shared<fa_interfaces::srv::ListDevices::Request>();
  request->refresh = true;
  auto future = client->async_send_request(request);

  ASSERT_TRUE(spinUntil(executor, [&future]() {
    return future.wait_for(0ms) == std::future_status::ready;
  }));
  const auto response = future.get();
  ASSERT_TRUE(response->success);
  ASSERT_EQ(response->device_ids.size(), 2u);
  ASSERT_EQ(response->max_input_channels.size(), 2u);
  ASSERT_EQ(response->default_sample_rates.size(), 2u);
  EXPECT_EQ(response->device_ids[0], "hw:0,0");
  EXPECT_EQ(response->max_input_channels[0], 2u);
  EXPECT_EQ(response->default_sample_rates[0], 44100u);
  EXPECT_EQ(response->device_ids[1], "hw:1,0");
  EXPECT_EQ(response->max_input_channels[1], 4u);
  EXPECT_EQ(response->default_sample_rates[1], 96000u);
  EXPECT_FALSE(node->hasFatalError());
}

TEST_F(RclcppContractTest, ReadFailureFailsClosedWithoutRetry)
{
  const auto state = std::make_shared<FakeSourceState>();
  state->fail_on_read.store(true);
  auto node = std::make_shared<fa_in::FaInNode>(
    optionsWith(validParameters()),
    factoryFor(state));

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  EXPECT_TRUE(spinUntil(executor, [&node]() {
    return node->hasFatalError();
  }));
  EXPECT_TRUE(node->hasFatalError());
  EXPECT_EQ(state->read_calls.load(), 1u);
}

TEST_F(RclcppContractTest, SwitchReopenFailureFailsClosed)
{
  const auto state = std::make_shared<FakeSourceState>();
  auto node = std::make_shared<fa_in::FaInNode>(
    optionsWith(validParameters()),
    factoryFor(state));
  auto client_node = std::make_shared<rclcpp::Node>("fa_in_contract_switch_client", quietContractNodeOptions());
  auto client = client_node->create_client<fa_interfaces::srv::SwitchDevice>("switch_device");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(client_node);

  ASSERT_TRUE(spinUntil(executor, [&client]() {
    return client->service_is_ready();
  }));
  state->fail_open_on_switch_target.store(true);
  auto request = std::make_shared<fa_interfaces::srv::SwitchDevice::Request>();
  request->target_selector_mode = "id";
  request->target_identifier = "hw:1,0";
  request->target_index = -1;
  client->async_send_request(request);

  EXPECT_TRUE(spinUntil(executor, [&node]() {
    return node->hasFatalError();
  }));
  EXPECT_TRUE(node->hasFatalError());
}

TEST_F(RclcppContractTest, SwitchDeviceReopensSelectedSource)
{
  const auto state = std::make_shared<FakeSourceState>();
  auto node = std::make_shared<fa_in::FaInNode>(
    optionsWith(validParameters()),
    factoryFor(state));
  auto client_node = std::make_shared<rclcpp::Node>("fa_in_contract_switch_success_client", quietContractNodeOptions());
  auto client = client_node->create_client<fa_interfaces::srv::SwitchDevice>("switch_device");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(client_node);

  ASSERT_TRUE(spinUntil(executor, [&client]() {
    return client->service_is_ready();
  }));
  auto request = std::make_shared<fa_interfaces::srv::SwitchDevice::Request>();
  request->target_selector_mode = "id";
  request->target_identifier = "hw:1,0";
  request->target_index = -1;
  auto future = client->async_send_request(request);

  ASSERT_TRUE(spinUntil(executor, [&future]() {
    return future.wait_for(0ms) == std::future_status::ready;
  }));
  const auto response = future.get();
  ASSERT_TRUE(response->success);
  EXPECT_EQ(response->message, "switched");
  EXPECT_TRUE(spinUntil(executor, [&state]() {
    return openedDevice(state) == "hw:1,0";
  }));
  EXPECT_FALSE(node->hasFatalError());
}

TEST_F(RclcppContractTest, SwitchDeviceDoesNotFallbackFromIdToDisplayName)
{
  const auto state = std::make_shared<FakeSourceState>();
  auto node = std::make_shared<fa_in::FaInNode>(
    optionsWith(validParameters()),
    factoryFor(state));
  auto client_node = std::make_shared<rclcpp::Node>("fa_in_contract_switch_id_miss_client", quietContractNodeOptions());
  auto client = client_node->create_client<fa_interfaces::srv::SwitchDevice>("switch_device");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(client_node);

  ASSERT_TRUE(spinUntil(executor, [&client]() {
    return client->service_is_ready();
  }));
  auto request = std::make_shared<fa_interfaces::srv::SwitchDevice::Request>();
  request->target_selector_mode = "id";
  request->target_identifier = "Backup Mic";
  request->target_index = -1;
  auto future = client->async_send_request(request);

  ASSERT_TRUE(spinUntil(executor, [&future]() {
    return future.wait_for(0ms) == std::future_status::ready;
  }));
  const auto response = future.get();
  ASSERT_FALSE(response->success);
  EXPECT_EQ(
    response->message,
    "switch_device target_identifier must be a raw hw: source id when target_selector_mode=id");
  EXPECT_EQ(openedDevice(state), "hw:0,0");
  EXPECT_FALSE(node->hasFatalError());
}

TEST_F(RclcppContractTest, SwitchDeviceUsesExplicitDisplayNameSelector)
{
  const auto state = std::make_shared<FakeSourceState>();
  auto node = std::make_shared<fa_in::FaInNode>(
    optionsWith(validParameters()),
    factoryFor(state));
  auto client_node = std::make_shared<rclcpp::Node>("fa_in_contract_switch_name_client", quietContractNodeOptions());
  auto client = client_node->create_client<fa_interfaces::srv::SwitchDevice>("switch_device");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(client_node);

  ASSERT_TRUE(spinUntil(executor, [&client]() {
    return client->service_is_ready();
  }));
  auto request = std::make_shared<fa_interfaces::srv::SwitchDevice::Request>();
  request->target_selector_mode = "name";
  request->target_identifier = "Backup Mic";
  request->target_index = -1;
  auto future = client->async_send_request(request);

  ASSERT_TRUE(spinUntil(executor, [&future]() {
    return future.wait_for(0ms) == std::future_status::ready;
  }));
  const auto response = future.get();
  ASSERT_TRUE(response->success);
  EXPECT_EQ(response->message, "switched");
  EXPECT_TRUE(spinUntil(executor, [&state]() {
    return openedDevice(state) == "hw:1,0";
  }));
  EXPECT_FALSE(node->hasFatalError());
}

}  // namespace

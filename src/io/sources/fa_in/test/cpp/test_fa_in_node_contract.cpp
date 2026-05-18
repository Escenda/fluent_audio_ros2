#include "fa_in/fa_in_node.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
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
  std::vector<uint8_t> payload{0, 0, 1, 0, 2, 0, 3, 0};
  std::mutex opened_device_mutex;
  std::string opened_device;
};

std::string displayName(const fa_in::backends::DeviceInfo & device)
{
  if (device.name.empty()) {
    return device.id;
  }
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
    if (selector.mode == "name") {
      for (const auto & device : devices) {
        if (device.id == selector.identifier || displayName(device) == selector.identifier) {
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

    std::fill(data, data + state_->payload.size(), uint8_t{0});
    std::copy(state_->payload.begin(), state_->payload.end(), data);
    std::this_thread::sleep_for(2ms);
    return fa_in::backends::ReadResult{fa_in::backends::ReadStatus::kOk, frames, ""};
  }

private:
  std::shared_ptr<FakeSourceState> state_;
  std::atomic<bool> open_{false};
};

std::vector<rclcpp::Parameter> validParameters()
{
  return {
    rclcpp::Parameter("backend.name", "alsa_capture"),
    rclcpp::Parameter("audio.device_selector.mode", "name"),
    rclcpp::Parameter("audio.device_selector.identifier", "hw:0,0"),
    rclcpp::Parameter("audio.device_selector.index", -1),
    rclcpp::Parameter("audio.sample_rate", 48000),
    rclcpp::Parameter("audio.channels", 1),
    rclcpp::Parameter("audio.bit_depth", 16),
    rclcpp::Parameter("audio.chunk_ms", 1),
    rclcpp::Parameter("audio.encoding", "PCM16LE"),
    rclcpp::Parameter("audio.stream_id", "audio/frame"),
    rclcpp::Parameter("audio.layout", "interleaved"),
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
  rclcpp::NodeOptions options;
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

TEST_F(RclcppContractTest, PublishesConfiguredMetadataAndRawSourcePayload)
{
  const auto state = std::make_shared<FakeSourceState>();
  auto node = std::make_shared<fa_in::FaInNode>(
    optionsWith(validParameters()),
    factoryFor(state));
  auto subscriber_node = std::make_shared<rclcpp::Node>("fa_in_contract_subscriber");
  std::optional<fa_interfaces::msg::AudioFrame> received;
  auto subscription = subscriber_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "audio/frame",
    rclcpp::SensorDataQoS(),
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
  EXPECT_EQ(received->stream_id, "audio/frame");
  EXPECT_EQ(received->encoding, "PCM16LE");
  EXPECT_EQ(received->sample_rate, 48000u);
  EXPECT_EQ(received->channels, 1u);
  EXPECT_EQ(received->bit_depth, 16u);
  EXPECT_EQ(received->layout, "interleaved");
  ASSERT_GE(received->data.size(), state->payload.size());
  EXPECT_TRUE(std::equal(state->payload.begin(), state->payload.end(), received->data.begin()));
  EXPECT_GE(state->read_calls.load(), 1u);
  EXPECT_FALSE(node->hasFatalError());
}

TEST_F(RclcppContractTest, ListDevicesSurfacesBackendEnumerationFailure)
{
  const auto state = std::make_shared<FakeSourceState>();
  auto node = std::make_shared<fa_in::FaInNode>(
    optionsWith(validParameters()),
    factoryFor(state));
  auto client_node = std::make_shared<rclcpp::Node>("fa_in_contract_list_devices_client");
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
  auto client_node = std::make_shared<rclcpp::Node>("fa_in_contract_list_capabilities_client");
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
  auto client_node = std::make_shared<rclcpp::Node>("fa_in_contract_switch_client");
  auto client = client_node->create_client<fa_interfaces::srv::SwitchDevice>("switch_device");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(client_node);

  ASSERT_TRUE(spinUntil(executor, [&client]() {
    return client->service_is_ready();
  }));
  state->fail_open_on_switch_target.store(true);
  auto request = std::make_shared<fa_interfaces::srv::SwitchDevice::Request>();
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
  auto client_node = std::make_shared<rclcpp::Node>("fa_in_contract_switch_success_client");
  auto client = client_node->create_client<fa_interfaces::srv::SwitchDevice>("switch_device");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(client_node);

  ASSERT_TRUE(spinUntil(executor, [&client]() {
    return client->service_is_ready();
  }));
  auto request = std::make_shared<fa_interfaces::srv::SwitchDevice::Request>();
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

}  // namespace

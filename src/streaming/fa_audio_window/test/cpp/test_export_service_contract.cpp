#include "fa_audio_window/fa_audio_window_node.hpp"

#include <chrono>
#include <filesystem>
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
using ArchiveAudioWindow = fa_interfaces::srv::ArchiveAudioWindow;
using ExportAudioWindow = fa_interfaces::srv::ExportAudioWindow;

constexpr const char * kInputTopic = "fa_audio_window/test/input";
constexpr const char * kServiceName = "fa_audio_window/test/export";
constexpr const char * kArchiveServiceName = "fa_audio_window/test/archive";
constexpr const char * kSourceId = "test-mic";
constexpr const char * kStreamId = "audio/test/mic";
constexpr int64_t kStartNs = 10000000000LL;

rclcpp::NodeOptions quietNodeOptions()
{
  rclcpp::NodeOptions options;
  options.start_parameter_services(false);
  options.start_parameter_event_publisher(false);
  options.enable_rosout(false);
  return options;
}

std::vector<rclcpp::Parameter> validParameters()
{
  return {
    rclcpp::Parameter("input_topic", kInputTopic),
    rclcpp::Parameter("service_name", kServiceName),
    rclcpp::Parameter("archive_service_name", kArchiveServiceName),
    rclcpp::Parameter("input.source_id", kSourceId),
    rclcpp::Parameter("input.stream_id", kStreamId),
    rclcpp::Parameter("expected.encoding", "PCM16LE"),
    rclcpp::Parameter("expected.sample_rate", 10),
    rclcpp::Parameter("expected.channels", 1),
    rclcpp::Parameter("expected.bit_depth", 16),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("window.retention_seconds", 2),
    rclcpp::Parameter("audio.default_scope", "mic"),
    rclcpp::Parameter("audio.supported_scopes", std::vector<std::string>{"mic"}),
    rclcpp::Parameter(
      "export.output_directory",
      (std::filesystem::temp_directory_path() / "fa_audio_window_service_test").string()),
    rclcpp::Parameter("export.codec", "pcm_s16le"),
    rclcpp::Parameter("export.container", "wav"),
    rclcpp::Parameter("export.payload_format", "audio/wav"),
    rclcpp::Parameter("window.id", "fa_audio_window_test"),
    rclcpp::Parameter("window.epoch", 1),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", false),
  };
}

rclcpp::NodeOptions optionsWith(std::vector<rclcpp::Parameter> parameters)
{
  rclcpp::NodeOptions options = quietNodeOptions();
  options.parameter_overrides(std::move(parameters));
  return options;
}

fa_interfaces::msg::AudioFrame validFrame()
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp.sec = static_cast<int32_t>(kStartNs / 1000000000LL);
  frame.header.stamp.nanosec = static_cast<uint32_t>(kStartNs % 1000000000LL);
  frame.source_id = kSourceId;
  frame.stream_id = kStreamId;
  frame.encoding = "PCM16LE";
  frame.sample_rate = 10u;
  frame.channels = 1u;
  frame.bit_depth = 16u;
  frame.layout = "interleaved";
  frame.epoch = 1u;
  frame.data = {0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00};
  return frame;
}

fa_interfaces::msg::AudioFrame validFrameAt(const int64_t start_unix_ns, const uint32_t epoch)
{
  fa_interfaces::msg::AudioFrame frame = validFrame();
  frame.header.stamp.sec = static_cast<int32_t>(start_unix_ns / 1000000000LL);
  frame.header.stamp.nanosec = static_cast<uint32_t>(start_unix_ns % 1000000000LL);
  frame.epoch = epoch;
  return frame;
}

std::shared_ptr<ExportAudioWindow::Request> requestFor(
  const std::string & time_range_spec,
  const std::string & scope,
  const std::string & codec = "pcm_s16le",
  const std::string & container = "wav",
  const std::string & payload_format = "audio/wav")
{
  auto request = std::make_shared<ExportAudioWindow::Request>();
  request->time_range_spec = time_range_spec;
  request->audio_scope = scope;
  request->codec = codec;
  request->container = container;
  request->payload_format = payload_format;
  return request;
}

std::shared_ptr<ArchiveAudioWindow::Request> archiveRequestFor(
  const std::string & time_range_spec,
  const std::string & scope,
  const std::string & reason,
  const std::string & codec = "pcm_s16le",
  const std::string & container = "wav",
  const std::string & payload_format = "audio/wav")
{
  auto request = std::make_shared<ArchiveAudioWindow::Request>();
  request->time_range_spec = time_range_spec;
  request->audio_scope = scope;
  request->reason = reason;
  request->related_artifact_ids = {"action_12"};
  request->codec = codec;
  request->container = container;
  request->payload_format = payload_format;
  return request;
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

class AudioWindowServiceContractTest : public ::testing::Test
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

TEST_F(AudioWindowServiceContractTest, ReturnsExplicitFailureContracts)
{
  auto node = std::make_shared<fa_audio_window::FaAudioWindowNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_audio_window_contract_io", quietNodeOptions());
  auto client = io_node->create_client<ExportAudioWindow>(kServiceName);
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic,
    rclcpp::QoS(10).best_effort());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&client]() {
    return client->service_is_ready();
  }));

  auto unresolved = client->async_send_request(requestFor("now-10s..now", "mic"));
  ASSERT_EQ(executor.spin_until_future_complete(unresolved, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto unresolved_response = unresolved.get();
  EXPECT_FALSE(unresolved_response->success);
  EXPECT_EQ(unresolved_response->error_code, ExportAudioWindow::Response::ERROR_TIME_RANGE_UNRESOLVED);

  auto empty = client->async_send_request(requestFor("10000000000..10100000000", "mic"));
  ASSERT_EQ(executor.spin_until_future_complete(empty, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto empty_response = empty.get();
  EXPECT_FALSE(empty_response->success);
  EXPECT_EQ(empty_response->error_code, ExportAudioWindow::Response::ERROR_WINDOW_NOT_FOUND);

  auto bad_scope = client->async_send_request(requestFor("10000000000..10100000000", "system"));
  ASSERT_EQ(executor.spin_until_future_complete(bad_scope, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto bad_scope_response = bad_scope.get();
  EXPECT_FALSE(bad_scope_response->success);
  EXPECT_EQ(bad_scope_response->error_code, ExportAudioWindow::Response::ERROR_UNSUPPORTED_AUDIO_SCOPE);

  auto bad_format = client->async_send_request(
    requestFor("10000000000..10100000000", "mic", "opus", "ogg", "audio/ogg"));
  ASSERT_EQ(executor.spin_until_future_complete(bad_format, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto bad_format_response = bad_format.get();
  EXPECT_FALSE(bad_format_response->success);
  EXPECT_EQ(bad_format_response->error_code, ExportAudioWindow::Response::ERROR_UNSUPPORTED_EXPORT_FORMAT);

  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0u;
  }));
  publisher->publish(validFrame());
  executor.spin_some(100ms);

  auto outside = client->async_send_request(requestFor("9000000000..10100000000", "mic"));
  ASSERT_EQ(executor.spin_until_future_complete(outside, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto outside_response = outside.get();
  EXPECT_FALSE(outside_response->success);
  EXPECT_EQ(outside_response->error_code, ExportAudioWindow::Response::ERROR_RANGE_OUTSIDE_WINDOW);

  auto success = client->async_send_request(requestFor("10000000000..10400000000", ""));
  ASSERT_EQ(executor.spin_until_future_complete(success, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto success_response = success.get();
  ASSERT_TRUE(success_response->success);
  EXPECT_EQ(success_response->error_code, ExportAudioWindow::Response::ERROR_NONE);
  EXPECT_FALSE(success_response->audio_clip_ref.clip_id.empty());
  EXPECT_FALSE(success_response->audio_clip_ref.uri.empty());
  EXPECT_EQ(success_response->audio_clip_ref.codec, "pcm_s16le");
  EXPECT_EQ(success_response->audio_clip_ref.container, "wav");
  EXPECT_EQ(success_response->audio_clip_ref.payload_format, "audio/wav");
  EXPECT_EQ(success_response->audio_clip_ref.sample_rate, 10u);
  EXPECT_EQ(success_response->audio_clip_ref.channels, 1u);
  EXPECT_EQ(success_response->audio_clip_ref.duration_ns, 400000000u);
  EXPECT_EQ(success_response->audio_clip_ref.time_range.start_unix_ns, 10000000000LL);
  EXPECT_EQ(success_response->audio_clip_ref.time_range.end_unix_ns, 10400000000LL);

  const std::string file_prefix = "file://";
  ASSERT_EQ(success_response->audio_clip_ref.uri.rfind(file_prefix, 0u), 0u);
  const std::filesystem::path exported_path = success_response->audio_clip_ref.uri.substr(file_prefix.size());
  EXPECT_TRUE(std::filesystem::exists(exported_path));

  publisher->publish(validFrameAt(11000000000LL, 2u));
  executor.spin_some(100ms);

  auto gap = client->async_send_request(requestFor("10000000000..11400000000", "mic"));
  ASSERT_EQ(executor.spin_until_future_complete(gap, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto gap_response = gap.get();
  EXPECT_FALSE(gap_response->success);
  EXPECT_EQ(gap_response->error_code, ExportAudioWindow::Response::ERROR_RANGE_OUTSIDE_WINDOW);
  EXPECT_EQ(gap_response->message, "requested time range is not continuously covered by retained audio");
}

TEST_F(AudioWindowServiceContractTest, RejectsDuplicateExportAndArchiveServiceNamesAtConstruction)
{
  std::vector<rclcpp::Parameter> parameters = validParameters();
  for (rclcpp::Parameter & parameter : parameters) {
    if (parameter.get_name() == "archive_service_name") {
      parameter = rclcpp::Parameter("archive_service_name", std::string{kServiceName});
    }
  }

  try {
    (void)std::make_shared<fa_audio_window::FaAudioWindowNode>(optionsWith(std::move(parameters)));
    FAIL() << "Expected duplicate service names to fail node construction";
  } catch (const std::runtime_error & e) {
    EXPECT_STREQ(e.what(), "service_name and archive_service_name must be different service names");
  }
}

TEST_F(AudioWindowServiceContractTest, ArchiveServiceReturnsClipOrExplicitErrors)
{
  auto node = std::make_shared<fa_audio_window::FaAudioWindowNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_audio_window_archive_io", quietNodeOptions());
  auto client = io_node->create_client<ArchiveAudioWindow>(kArchiveServiceName);
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic,
    rclcpp::QoS(10).best_effort());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&client]() {
    return client->service_is_ready();
  }));

  auto empty_reason = client->async_send_request(archiveRequestFor("10000000000..10100000000", "mic", ""));
  ASSERT_EQ(executor.spin_until_future_complete(empty_reason, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto empty_reason_response = empty_reason.get();
  EXPECT_FALSE(empty_reason_response->success);
  EXPECT_EQ(
    empty_reason_response->error_code,
    ArchiveAudioWindow::Response::ERROR_INVALID_ARCHIVE_REQUEST);

  auto unresolved = client->async_send_request(archiveRequestFor("now-10s..now", "mic", "operator evidence"));
  ASSERT_EQ(executor.spin_until_future_complete(unresolved, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto unresolved_response = unresolved.get();
  EXPECT_FALSE(unresolved_response->success);
  EXPECT_EQ(unresolved_response->error_code, ArchiveAudioWindow::Response::ERROR_TIME_RANGE_UNRESOLVED);

  auto bad_scope = client->async_send_request(archiveRequestFor("10000000000..10100000000", "system", "evidence"));
  ASSERT_EQ(executor.spin_until_future_complete(bad_scope, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto bad_scope_response = bad_scope.get();
  EXPECT_FALSE(bad_scope_response->success);
  EXPECT_EQ(bad_scope_response->error_code, ArchiveAudioWindow::Response::ERROR_UNSUPPORTED_AUDIO_SCOPE);

  auto bad_format = client->async_send_request(
    archiveRequestFor("10000000000..10100000000", "mic", "evidence", "opus", "ogg", "audio/ogg"));
  ASSERT_EQ(executor.spin_until_future_complete(bad_format, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto bad_format_response = bad_format.get();
  EXPECT_FALSE(bad_format_response->success);
  EXPECT_EQ(bad_format_response->error_code, ArchiveAudioWindow::Response::ERROR_UNSUPPORTED_ARCHIVE_FORMAT);

  auto empty_window = client->async_send_request(archiveRequestFor("10000000000..10100000000", "mic", "evidence"));
  ASSERT_EQ(executor.spin_until_future_complete(empty_window, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto empty_window_response = empty_window.get();
  EXPECT_FALSE(empty_window_response->success);
  EXPECT_EQ(empty_window_response->error_code, ArchiveAudioWindow::Response::ERROR_WINDOW_NOT_FOUND);

  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0u;
  }));
  publisher->publish(validFrame());
  executor.spin_some(100ms);

  auto success = client->async_send_request(
    archiveRequestFor("10000000000..10400000000", "", "operator evidence"));
  ASSERT_EQ(executor.spin_until_future_complete(success, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto success_response = success.get();
  ASSERT_TRUE(success_response->success);
  EXPECT_EQ(success_response->error_code, ArchiveAudioWindow::Response::ERROR_NONE);
  EXPECT_FALSE(success_response->audio_clip_ref.clip_id.empty());
  EXPECT_FALSE(success_response->audio_clip_ref.uri.empty());
  EXPECT_EQ(success_response->audio_clip_ref.codec, "pcm_s16le");
  EXPECT_EQ(success_response->audio_clip_ref.container, "wav");
  EXPECT_EQ(success_response->audio_clip_ref.payload_format, "audio/wav");
  EXPECT_EQ(success_response->audio_clip_ref.duration_ns, 400000000u);
  EXPECT_EQ(success_response->time_range.start_unix_ns, 10000000000LL);
  EXPECT_EQ(success_response->time_range.end_unix_ns, 10400000000LL);

  const std::string file_prefix = "file://";
  ASSERT_EQ(success_response->audio_clip_ref.uri.rfind(file_prefix, 0u), 0u);
  const std::filesystem::path archived_path = success_response->audio_clip_ref.uri.substr(file_prefix.size());
  EXPECT_TRUE(std::filesystem::exists(archived_path));

  publisher->publish(validFrameAt(11000000000LL, 2u));
  executor.spin_some(100ms);

  auto gap = client->async_send_request(
    archiveRequestFor("10000000000..11400000000", "mic", "operator evidence"));
  ASSERT_EQ(executor.spin_until_future_complete(gap, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto gap_response = gap.get();
  EXPECT_FALSE(gap_response->success);
  EXPECT_EQ(gap_response->error_code, ArchiveAudioWindow::Response::ERROR_RANGE_NOT_CONTINUOUS);
}
}  // namespace

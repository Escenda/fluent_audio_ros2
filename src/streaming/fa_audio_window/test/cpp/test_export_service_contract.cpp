#include "fa_audio_window/fa_audio_window_node.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
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

std::string sanitizeTestName(const std::string & value)
{
  std::string result;
  result.reserve(value.size());
  for (const char c : value) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-') {
      result.push_back(c);
    } else {
      result.push_back('_');
    }
  }
  return result.empty() ? "unnamed_test" : result;
}

std::filesystem::path outputDirectoryForCurrentTest()
{
  const ::testing::TestInfo * const test_info =
    ::testing::UnitTest::GetInstance()->current_test_info();
  if (test_info == nullptr) {
    throw std::runtime_error("current gtest info is unavailable");
  }
  return std::filesystem::temp_directory_path() / "fa_audio_window_service_test" /
    sanitizeTestName(std::string{test_info->test_suite_name()} + "_" + test_info->name());
}

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
    rclcpp::Parameter("export.output_directory", outputDirectoryForCurrentTest().string()),
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
  const std::string & payload_format = "audio/wav",
  const std::vector<std::string> & related_artifact_ids =
    std::vector<std::string>{"action_12"})
{
  auto request = std::make_shared<ArchiveAudioWindow::Request>();
  request->time_range_spec = time_range_spec;
  request->audio_scope = scope;
  request->reason = reason;
  request->related_artifact_ids = related_artifact_ids;
  request->codec = codec;
  request->container = container;
  request->payload_format = payload_format;
  return request;
}

std::string readTextFile(const std::filesystem::path & path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open test file");
  }
  std::ostringstream content;
  content << in.rdbuf();
  return content.str();
}

std::vector<uint8_t> readBinaryFile(const std::filesystem::path & path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open test file");
  }
  return {
    std::istreambuf_iterator<char>(in),
    std::istreambuf_iterator<char>(),
  };
}

std::filesystem::path pathFromClipRef(const fa_interfaces::msg::AudioClipRef & ref)
{
  const std::string file_prefix = "file://";
  if (ref.uri.rfind(file_prefix, 0u) != 0u) {
    throw std::runtime_error("clip ref URI is not a file URI");
  }
  return ref.uri.substr(file_prefix.size());
}

size_t countWavFiles(const std::filesystem::path & directory)
{
  size_t count = 0u;
  for (const std::filesystem::directory_entry & entry : std::filesystem::directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().extension() == ".wav") {
      ++count;
    }
  }
  return count;
}

size_t countTemporaryPublishFiles(const std::filesystem::path & directory)
{
  size_t count = 0u;
  for (const std::filesystem::directory_entry & entry : std::filesystem::directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().filename().string().find(".tmp") != std::string::npos) {
      ++count;
    }
  }
  return count;
}

void expectSameClipRef(
  const fa_interfaces::msg::AudioClipRef & left,
  const fa_interfaces::msg::AudioClipRef & right)
{
  EXPECT_EQ(left.clip_id, right.clip_id);
  EXPECT_EQ(left.uri, right.uri);
  EXPECT_EQ(left.codec, right.codec);
  EXPECT_EQ(left.container, right.container);
  EXPECT_EQ(left.payload_format, right.payload_format);
  EXPECT_EQ(left.sample_rate, right.sample_rate);
  EXPECT_EQ(left.channels, right.channels);
  EXPECT_EQ(left.duration_ns, right.duration_ns);
  EXPECT_EQ(left.time_range.start_unix_ns, right.time_range.start_unix_ns);
  EXPECT_EQ(left.time_range.end_unix_ns, right.time_range.end_unix_ns);
  EXPECT_EQ(left.time_range.clock, right.time_range.clock);
  EXPECT_EQ(left.time_range.uncertainty_ns, right.time_range.uncertainty_ns);
  EXPECT_EQ(left.time_range.uncertainty_reason, right.time_range.uncertainty_reason);
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
    std::filesystem::remove_all(outputDirectoryForCurrentTest());
    std::filesystem::create_directories(outputDirectoryForCurrentTest());
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

TEST_F(AudioWindowServiceContractTest, RepeatedExportRequestReturnsDeterministicClipRef)
{
  auto node = std::make_shared<fa_audio_window::FaAudioWindowNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_audio_window_export_idempotency_io", quietNodeOptions());
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0u;
  }));
  publisher->publish(validFrame());
  executor.spin_some(100ms);

  auto first = client->async_send_request(requestFor("10000000000..10400000000", ""));
  ASSERT_EQ(executor.spin_until_future_complete(first, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto first_response = first.get();
  ASSERT_TRUE(first_response->success);

  auto second = client->async_send_request(requestFor("10000000000..10400000000", ""));
  ASSERT_EQ(executor.spin_until_future_complete(second, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto second_response = second.get();
  ASSERT_TRUE(second_response->success);

  expectSameClipRef(first_response->audio_clip_ref, second_response->audio_clip_ref);
  EXPECT_EQ(first_response->time_range.start_unix_ns, second_response->time_range.start_unix_ns);
  EXPECT_EQ(first_response->time_range.end_unix_ns, second_response->time_range.end_unix_ns);
  EXPECT_TRUE(std::filesystem::exists(pathFromClipRef(first_response->audio_clip_ref)));
  EXPECT_EQ(countWavFiles(outputDirectoryForCurrentTest()), 1u);
}

TEST_F(AudioWindowServiceContractTest, RejectsPreexistingDeterministicExportConflict)
{
  auto node = std::make_shared<fa_audio_window::FaAudioWindowNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_audio_window_export_conflict_io", quietNodeOptions());
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0u;
  }));
  publisher->publish(validFrame());
  executor.spin_some(100ms);

  auto first = client->async_send_request(requestFor("10000000000..10400000000", "mic"));
  ASSERT_EQ(executor.spin_until_future_complete(first, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto first_response = first.get();
  ASSERT_TRUE(first_response->success);
  const std::filesystem::path exported_path = pathFromClipRef(first_response->audio_clip_ref);
  {
    std::ofstream out(exported_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out << "corrupt wav";
  }

  auto conflict = client->async_send_request(requestFor("10000000000..10400000000", "mic"));
  ASSERT_EQ(executor.spin_until_future_complete(conflict, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto conflict_response = conflict.get();
  EXPECT_FALSE(conflict_response->success);
  EXPECT_EQ(conflict_response->error_code, ExportAudioWindow::Response::ERROR_EXPORT_FAILED);
  EXPECT_EQ(conflict_response->message, "deterministic audio clip path exists with different bytes");
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

TEST_F(AudioWindowServiceContractTest, RepeatedArchiveRequestReturnsDeterministicClipBytesAndMetadata)
{
  auto node = std::make_shared<fa_audio_window::FaAudioWindowNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_audio_window_archive_idempotency_io", quietNodeOptions());
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0u;
  }));
  publisher->publish(validFrame());
  executor.spin_some(100ms);

  auto first = client->async_send_request(
    archiveRequestFor(
      "10000000000..10400000000",
      "mic",
      "operator evidence",
      "pcm_s16le",
      "wav",
      "audio/wav",
      std::vector<std::string>{"action_12", "video_observation_9"}));
  ASSERT_EQ(executor.spin_until_future_complete(first, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto first_response = first.get();
  ASSERT_TRUE(first_response->success);
  const std::filesystem::path first_path = pathFromClipRef(first_response->audio_clip_ref);
  const std::vector<uint8_t> first_bytes = readBinaryFile(first_path);
  const std::string first_metadata = readTextFile(first_path.string() + ".metadata.json");

  auto second = client->async_send_request(
    archiveRequestFor(
      "10000000000..10400000000",
      "mic",
      "operator evidence",
      "pcm_s16le",
      "wav",
      "audio/wav",
      std::vector<std::string>{"action_12", "video_observation_9"}));
  ASSERT_EQ(executor.spin_until_future_complete(second, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto second_response = second.get();
  ASSERT_TRUE(second_response->success);
  const std::filesystem::path second_path = pathFromClipRef(second_response->audio_clip_ref);

  expectSameClipRef(first_response->audio_clip_ref, second_response->audio_clip_ref);
  EXPECT_EQ(first_path, second_path);
  EXPECT_EQ(first_bytes, readBinaryFile(second_path));
  EXPECT_EQ(first_metadata, readTextFile(second_path.string() + ".metadata.json"));
  EXPECT_EQ(countWavFiles(outputDirectoryForCurrentTest()), 1u);
}

TEST_F(AudioWindowServiceContractTest, RecreatesMissingDeterministicArchiveMetadata)
{
  auto node = std::make_shared<fa_audio_window::FaAudioWindowNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_audio_window_archive_metadata_recreate_io", quietNodeOptions());
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0u;
  }));
  publisher->publish(validFrame());
  executor.spin_some(100ms);

  auto first = client->async_send_request(
    archiveRequestFor("10000000000..10400000000", "mic", "metadata recreation evidence"));
  ASSERT_EQ(executor.spin_until_future_complete(first, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto first_response = first.get();
  ASSERT_TRUE(first_response->success);
  const std::filesystem::path clip_path = pathFromClipRef(first_response->audio_clip_ref);
  const std::filesystem::path metadata_path = clip_path.string() + ".metadata.json";
  const std::vector<uint8_t> first_bytes = readBinaryFile(clip_path);
  ASSERT_TRUE(std::filesystem::remove(metadata_path));
  ASSERT_FALSE(std::filesystem::exists(metadata_path));

  auto second = client->async_send_request(
    archiveRequestFor("10000000000..10400000000", "mic", "metadata recreation evidence"));
  ASSERT_EQ(executor.spin_until_future_complete(second, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto second_response = second.get();
  ASSERT_TRUE(second_response->success);

  expectSameClipRef(first_response->audio_clip_ref, second_response->audio_clip_ref);
  EXPECT_EQ(first_bytes, readBinaryFile(clip_path));
  EXPECT_TRUE(std::filesystem::exists(metadata_path));
  EXPECT_EQ(countWavFiles(outputDirectoryForCurrentTest()), 1u);
  EXPECT_EQ(countTemporaryPublishFiles(outputDirectoryForCurrentTest()), 0u);
}

TEST_F(AudioWindowServiceContractTest, RejectsPreexistingDeterministicArchiveConflicts)
{
  auto node = std::make_shared<fa_audio_window::FaAudioWindowNode>(optionsWith(validParameters()));
  auto io_node = std::make_shared<rclcpp::Node>("fa_audio_window_archive_conflict_io", quietNodeOptions());
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
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0u;
  }));
  publisher->publish(validFrame());
  executor.spin_some(100ms);

  auto media_success = client->async_send_request(
    archiveRequestFor("10000000000..10400000000", "mic", "media conflict evidence"));
  ASSERT_EQ(executor.spin_until_future_complete(media_success, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto media_success_response = media_success.get();
  ASSERT_TRUE(media_success_response->success);
  const std::filesystem::path media_path = pathFromClipRef(media_success_response->audio_clip_ref);
  {
    std::ofstream out(media_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out << "corrupt wav";
  }

  auto media_conflict = client->async_send_request(
    archiveRequestFor("10000000000..10400000000", "mic", "media conflict evidence"));
  ASSERT_EQ(executor.spin_until_future_complete(media_conflict, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto media_conflict_response = media_conflict.get();
  EXPECT_FALSE(media_conflict_response->success);
  EXPECT_EQ(media_conflict_response->error_code, ArchiveAudioWindow::Response::ERROR_ARCHIVE_FAILED);
  EXPECT_EQ(media_conflict_response->message, "deterministic audio clip path exists with different bytes");

  auto metadata_success = client->async_send_request(
    archiveRequestFor("10000000000..10400000000", "mic", "metadata conflict evidence"));
  ASSERT_EQ(executor.spin_until_future_complete(metadata_success, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto metadata_success_response = metadata_success.get();
  ASSERT_TRUE(metadata_success_response->success);
  const std::filesystem::path metadata_clip_path = pathFromClipRef(metadata_success_response->audio_clip_ref);
  const std::filesystem::path metadata_path = metadata_clip_path.string() + ".metadata.json";
  {
    std::ofstream out(metadata_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out << "{\"schema\":\"corrupt\"}\n";
  }

  auto metadata_conflict = client->async_send_request(
    archiveRequestFor("10000000000..10400000000", "mic", "metadata conflict evidence"));
  ASSERT_EQ(executor.spin_until_future_complete(metadata_conflict, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const auto metadata_conflict_response = metadata_conflict.get();
  EXPECT_FALSE(metadata_conflict_response->success);
  EXPECT_EQ(metadata_conflict_response->error_code, ArchiveAudioWindow::Response::ERROR_ARCHIVE_FAILED);
  EXPECT_EQ(
    metadata_conflict_response->message,
    "deterministic archive metadata path exists with different content");
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
    archiveRequestFor(
      "10000000000..10400000000",
      "",
      " operator \"evidence\" \\ line\nnext\tend ",
      "pcm_s16le",
      "wav",
      "audio/wav",
      std::vector<std::string>{"action_12", "video\\observation\n9"}));
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
  const std::filesystem::path metadata_path = archived_path.string() + ".metadata.json";
  ASSERT_TRUE(std::filesystem::exists(metadata_path));
  EXPECT_EQ(countTemporaryPublishFiles(outputDirectoryForCurrentTest()), 0u);
  const nlohmann::json metadata = nlohmann::json::parse(readTextFile(metadata_path));
  EXPECT_EQ(metadata.at("schema").get<std::string>(), "fluent_audio.archive_metadata.v1");
  EXPECT_EQ(metadata.at("operation").get<std::string>(), "archive_audio_window");
  EXPECT_EQ(metadata.at("reason").get<std::string>(), "operator \"evidence\" \\ line\nnext\tend");
  ASSERT_TRUE(metadata.at("related_artifact_ids").is_array());
  ASSERT_EQ(metadata.at("related_artifact_ids").size(), 2u);
  EXPECT_EQ(metadata.at("related_artifact_ids").at(0).get<std::string>(), "action_12");
  EXPECT_EQ(metadata.at("related_artifact_ids").at(1).get<std::string>(), "video\\observation\n9");
  EXPECT_EQ(metadata.at("source_id").get<std::string>(), "test-mic");
  EXPECT_EQ(metadata.at("stream_id").get<std::string>(), "audio/test/mic");
  EXPECT_EQ(metadata.at("window_id").get<std::string>(), "fa_audio_window_test");
  EXPECT_EQ(metadata.at("window_epoch").get<uint64_t>(), 1u);
  EXPECT_EQ(metadata.at("audio_scope").get<std::string>(), "mic");

  const nlohmann::json time_range = metadata.at("time_range");
  EXPECT_EQ(time_range.at("start_unix_ns").get<int64_t>(), 10000000000LL);
  EXPECT_EQ(time_range.at("end_unix_ns").get<int64_t>(), 10400000000LL);
  EXPECT_EQ(time_range.at("clock").get<std::string>(), "media");
  EXPECT_EQ(time_range.at("uncertainty_ns").get<uint64_t>(), 0u);
  EXPECT_EQ(time_range.at("uncertainty_reason").get<std::string>(), "");

  const nlohmann::json clip_ref = metadata.at("audio_clip_ref");
  EXPECT_EQ(clip_ref.at("clip_id").get<std::string>(), success_response->audio_clip_ref.clip_id);
  EXPECT_EQ(clip_ref.at("uri").get<std::string>(), success_response->audio_clip_ref.uri);
  EXPECT_EQ(clip_ref.at("codec").get<std::string>(), "pcm_s16le");
  EXPECT_EQ(clip_ref.at("container").get<std::string>(), "wav");
  EXPECT_EQ(clip_ref.at("payload_format").get<std::string>(), "audio/wav");
  EXPECT_EQ(clip_ref.at("sample_rate").get<uint32_t>(), 10u);
  EXPECT_EQ(clip_ref.at("channels").get<uint32_t>(), 1u);
  EXPECT_EQ(clip_ref.at("duration_ns").get<uint64_t>(), 400000000u);

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

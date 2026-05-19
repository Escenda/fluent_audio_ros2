#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "fa_audio_window/audio_format.hpp"
#include "fa_audio_window/audio_window_buffer.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_interfaces/srv/archive_audio_window.hpp"
#include "fa_interfaces/srv/export_audio_window.hpp"

namespace fa_audio_window
{

struct AudioWindowConfig
{
  std::string input_topic{};
  std::string service_name{};
  std::string archive_service_name{};
  std::string source_id{};
  std::string stream_id{};
  AudioFormat expected_format{};
  std::string default_audio_scope{};
  std::set<std::string> supported_audio_scopes{};
  uint64_t retention_ns{0};
  std::filesystem::path output_directory{};
  std::string supported_codec{};
  std::string supported_container{};
  std::string supported_payload_format{};
  std::string window_id{};
  uint64_t window_epoch{1};
  int qos_depth{0};
  bool qos_reliable{false};
};

class FaAudioWindowNode : public rclcpp::Node
{
public:
  explicit FaAudioWindowNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaAudioWindowNode() override = default;

private:
  enum class ClipOperationError
  {
    kNone,
    kTimeRangeUnresolved,
    kWindowNotFound,
    kRangeOutsideWindow,
    kRangeNotContinuous,
    kNoSamplesSelected,
    kUnsupportedAudioScope,
    kUnsupportedFormat,
    kWriteFailed,
  };

  struct ClipOperationRequest
  {
    std::string time_range_spec{};
    std::string audio_scope{};
    std::string codec{};
    std::string container{};
    std::string payload_format{};
    std::string operation_name{};
    std::string archive_reason{};
    std::vector<std::string> related_artifact_ids{};
  };

  struct ClipOperationResult
  {
    bool success{false};
    ClipOperationError error{ClipOperationError::kNone};
    std::string message{};
    fa_interfaces::msg::AudioClipRef audio_clip_ref{};
    fa_interfaces::msg::ResolvedTimeRange time_range{};
  };

  using ArchiveAudioWindow = fa_interfaces::srv::ArchiveAudioWindow;
  using ExportAudioWindow = fa_interfaces::srv::ExportAudioWindow;

  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void handleExportRequest(
    const std::shared_ptr<ExportAudioWindow::Request> request,
    std::shared_ptr<ExportAudioWindow::Response> response);
  void handleArchiveRequest(
    const std::shared_ptr<ArchiveAudioWindow::Request> request,
    std::shared_ptr<ArchiveAudioWindow::Response> response);

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  TimedAudioFrame toTimedAudioFrame(const fa_interfaces::msg::AudioFrame & msg) const;
  std::string resolveAudioScope(const std::string & requested_scope) const;
  bool isSupportedFormat(
    const std::string & codec,
    const std::string & container,
    const std::string & payload_format) const;
  ClipOperationResult writeWindowClip(const ClipOperationRequest & request);
  void setError(
    ExportAudioWindow::Response & response,
    const std::string & error_code,
    const std::string & message) const;
  void setError(
    ArchiveAudioWindow::Response & response,
    const std::string & error_code,
    const std::string & message) const;
  std::string exportErrorCode(ClipOperationError error) const;
  std::string archiveErrorCode(ClipOperationError error) const;
  void fillResolvedRange(
    fa_interfaces::msg::ResolvedTimeRange & msg,
    const TimeRange & range) const;
  void fillAudioClipRef(
    fa_interfaces::msg::AudioClipRef & ref,
    const std::string & clip_id,
    const std::filesystem::path & path,
    const WindowQueryResult & query) const;
  std::string clipIdFor(
    const ClipOperationRequest & request,
    const std::string & resolved_scope,
    const TimeRange & exported_range) const;
  std::filesystem::path clipPathFor(const std::string & clip_id) const;
  std::string archiveMetadataJson(
    const ClipOperationRequest & request,
    const fa_interfaces::msg::AudioClipRef & clip_ref,
    const TimeRange & exported_range) const;
  void writeTextAtomically(
    const std::filesystem::path & path,
    const std::string & content) const;

  AudioWindowConfig config_{};
  std::unique_ptr<AudioWindowBuffer> buffer_{};
  mutable std::mutex buffer_mutex_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Service<ExportAudioWindow>::SharedPtr export_service_;
  rclcpp::Service<ArchiveAudioWindow>::SharedPtr archive_service_;
  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_buffered_{0};
  std::atomic<uint64_t> frames_dropped_{0};
};

}  // namespace fa_audio_window

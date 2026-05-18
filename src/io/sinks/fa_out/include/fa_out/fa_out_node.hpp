#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_interfaces/msg/playback_done.hpp"
#include "fa_interfaces/srv/playback_control.hpp"
#include "fa_out/backends/sink_backend.hpp"
#include "std_msgs/msg/header.hpp"

namespace fa_out
{

namespace backends
{
struct AlsaPlaybackConfig;
}

struct OutputConfig
{
  std::string backend_name{};
  std::string input_topic{};
  std::string input_stream_id{};
  std::string playback_done_topic{};
  std::string playback_control_service{};
  std::string device_id{};
  std::string encoding{};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  uint32_t bit_depth{0};
  size_t max_queue_frames{0};
  uint32_t chunk_duration_ms{0};
  size_t playback_chunk_frames{0};
  size_t playback_chunk_bytes{0};
  size_t alsa_buffer_frames{0};
  size_t alsa_period_frames{0};
  size_t qos_depth{0};
  bool qos_reliable{false};
  size_t lifecycle_qos_depth{0};
  bool lifecycle_qos_reliable{false};
};

struct QueuedFrame
{
  std_msgs::msg::Header header;
  std::string request_id{};
  uint32_t epoch{0};
  std::vector<uint8_t> data;
};

class FaOutNode : public rclcpp::Node
{
public:
  using BackendFactory = std::function<
    std::unique_ptr<backends::SinkBackend>(const backends::AlsaPlaybackConfig &)>;

  explicit FaOutNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  FaOutNode(const rclcpp::NodeOptions & options, BackendFactory backend_factory);
  ~FaOutNode() override;
  bool hasFatalError() const;

private:
  void loadParameters();
  void openBackend();
  void closeBackend();
  size_t writeBackendFrames(const uint8_t * data, size_t frame_count);
  bool isBackendRunning();
  void failClosed(const std::string &reason);
  void playbackThread();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void handlePlaybackControl(
    const std::shared_ptr<fa_interfaces::srv::PlaybackControl::Request> request,
    std::shared_ptr<fa_interfaces::srv::PlaybackControl::Response> response);
  void publishPlaybackDone(const QueuedFrame & queued_frame);
  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);

  OutputConfig config_;
  BackendFactory backend_factory_;
  std::unique_ptr<backends::SinkBackend> sink_backend_;
  std::mutex backend_mutex_;
  size_t bytes_per_frame_ = 0;

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<QueuedFrame> frame_queue_;
  std::thread playback_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> fatal_error_{false};
  std::atomic<bool> paused_{false};
  std::atomic<uint32_t> active_epoch_{0};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::PlaybackDone>::SharedPtr playback_done_pub_;
  rclcpp::Service<fa_interfaces::srv::PlaybackControl>::SharedPtr playback_control_srv_;
};

}  // namespace fa_out

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_interfaces/msg/playback_done.hpp"
#include "fa_out/backends/sink_backend.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/header.hpp"

namespace fa_out
{

struct OutputConfig
{
  std::string backend_name{};
  std::string device_id{};
  std::string encoding{};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  uint32_t bit_depth{0};
  size_t max_queue_frames{0};
  uint32_t chunk_duration_ms{0};
  size_t qos_depth{0};
  bool qos_reliable{false};
};

// 再生キュー用の構造体（フレームデータとヘッダーをペアで保持）
struct QueuedFrame
{
  std::vector<uint8_t> data;
  std_msgs::msg::Header header;
  uint32_t epoch{0};
};

class FaOutNode : public rclcpp::Node
{
public:
  FaOutNode();
  ~FaOutNode() override;
  bool hasFatalError() const;

private:
  void loadParameters();
  void openBackend();
  void closeBackend();
  bool discardBackendBuffer(const char *operation);
  size_t writeBackendFrames(const uint8_t * data, size_t frame_count);
  bool isBackendRunning();
  void failClosed(const std::string &reason);
  void playbackThread();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void handleStop(const std_msgs::msg::Empty::SharedPtr msg);
  void handlePause(const std_msgs::msg::Empty::SharedPtr msg);
  void handleResume(const std_msgs::msg::Empty::SharedPtr msg);
  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);

  OutputConfig config_;
  std::unique_ptr<backends::SinkBackend> sink_backend_;
  std::mutex backend_mutex_;
  size_t bytes_per_frame_ = 0;

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<QueuedFrame> frame_queue_;
  std::thread playback_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> fatal_error_{false};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr stop_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr pause_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr resume_sub_;
  rclcpp::Publisher<fa_interfaces::msg::PlaybackDone>::SharedPtr playback_done_pub_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr paused_pub_;  // 一時停止完了通知

  std::atomic<bool> stop_requested_{false};  // 停止リクエストフラグ
  std::atomic<bool> pause_requested_{false};  // 一時停止リクエストフラグ
  std::atomic<bool> is_paused_{false};  // 一時停止中フラグ

  // Barge-in 後の古いフレームをフィルタリングするためのタイムスタンプ
  std::mutex stop_time_mutex_;
  rclcpp::Time last_stop_time_{0, 0, RCL_ROS_TIME};

  // stop 世代（epoch）。
  // stop を受けたらインクリメントし、古い epoch のフレームは破棄する。
  std::atomic<uint32_t> current_epoch_{1};
};

}  // namespace fa_out

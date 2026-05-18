#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_monitor_mix
{

namespace backends
{
class InternalMonitorMixBackend;
}  // namespace backends

struct MonitorMixConfig
{
  std::vector<std::string> input_topics{"audio/program/frame", "audio/tts/frame"};
  std::vector<std::string> input_stream_ids{"audio/program/frame", "audio/tts/frame"};
  std::vector<double> input_gains_db{0.0};
  std::vector<double> input_gains_linear{1.0};
  int master_index{0};
  std::string output_topic{"audio/monitor/frame"};
  std::string output_stream_id{"audio/monitor/frame"};
  std::string output_source_id{"monitor_mix"};
  int expected_sample_rate{48000};
  int expected_channels{2};
  std::string expected_encoding{"FLOAT32LE"};
  int expected_bit_depth{32};
  std::string expected_layout{"interleaved"};
  int max_frame_age_ms{100};
  int qos_depth{10};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{1000};
};

/**
 * @brief 複数の AudioFrame stream から fail-closed な monitor bus mix を生成する routing node。
 *
 * @details device 出力、resample、limiter、normalize は行わない。必要な入力 frame が揃わない場合や
 * sample 範囲を超える場合は monitor frame を publish せず drop として数える。
 */
class FaMonitorMixNode : public rclcpp::Node
{
public:
  explicit FaMonitorMixNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaMonitorMixNode() override;

private:
  void loadParameters();
  void configureBackend();
  void setupInterfaces();
  void handleInputFrame(size_t index, const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(
    const fa_interfaces::msg::AudioFrame & msg,
    const std::string & expected_stream_id);
  bool mixAndPublish(const fa_interfaces::msg::AudioFrame & master_frame);
  double gainDbForIndex(size_t index) const;

  MonitorMixConfig config_;
  std::unique_ptr<backends::InternalMonitorMixBackend> backend_;
  std::vector<rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr> input_subs_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  mutable std::mutex frames_mutex_;
  std::vector<fa_interfaces::msg::AudioFrame::SharedPtr> latest_frames_;
  std::vector<rclcpp::Time> latest_frame_received_at_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_valid_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> mix_frames_out_{0};
  std::atomic<uint64_t> mix_frames_dropped_{0};
  std::atomic<uint64_t> stale_frame_drops_{0};
  std::atomic<uint64_t> missing_frame_drops_{0};
  std::atomic<uint64_t> range_drops_{0};
};

}  // namespace fa_monitor_mix

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_sidechain/backends/internal_sidechain_detector.hpp"

namespace fa_sidechain
{

struct SidechainConfig
{
  std::string sidechain_topic{};
  std::string control_topic{};
  std::string sidechain_stream_id{};
  std::string control_stream_id{};
  double threshold_rms{-1.0};
  double active_gain_db{0.0};
  double inactive_gain_db{0.0};
  int control_sample_rate{-1};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

/**
 * @brief Sidechain AudioFrame を明示的な mono gain-control AudioFrame へ変換する routing node。
 *
 * @details 入力波形の metadata validation と ROS publication を担当する。
 * RMS / gain 決定 / control data bytes の生成は ROS-free backend に閉じる。
 */
class FaSidechainNode : public rclcpp::Node
{
public:
  explicit FaSidechainNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaSidechainNode() override = default;

private:
  void loadParameters();
  void configureBackend();
  void setupInterfaces();
  void handleSidechainFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg) const;
  bool buildControlFrame(
    const fa_interfaces::msg::AudioFrame & input,
    const std::vector<uint8_t> & control_data,
    fa_interfaces::msg::AudioFrame & output) const;
  size_t bytesPerFrame() const;

  SidechainConfig config_;
  std::unique_ptr<backends::InternalSidechainDetectorBackend> backend_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr sidechain_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr control_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> active_frames_{0};
  std::atomic<uint64_t> inactive_frames_{0};
};

}  // namespace fa_sidechain

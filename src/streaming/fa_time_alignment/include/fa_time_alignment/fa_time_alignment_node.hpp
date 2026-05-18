#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_time_alignment
{

struct TimeAlignmentConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string input_stream_id{};
  std::string output_stream_id{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  double alignment_period_ms{-1.0};
  double alignment_phase_ms{-1.0};
  double alignment_max_adjust_ms{-1.0};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

class FaTimeAlignmentNode : public rclcpp::Node
{
public:
  explicit FaTimeAlignmentNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaTimeAlignmentNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool alignFrame(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  TimeAlignmentConfig config_;

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> frames_aligned_{0};
  std::atomic<uint64_t> frames_excess_adjust_{0};
};

}  // namespace fa_time_alignment

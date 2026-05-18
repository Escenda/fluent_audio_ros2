#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_latency_compensation
{

struct LatencyCompensationConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string input_stream_id{};
  std::string output_stream_id{};
  double offset_ms{0.0};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

/**
 * @brief AudioFrame の byte payload を変更せず header.stamp だけを補正する streaming node。
 */
class FaLatencyCompensationNode : public rclcpp::Node
{
public:
  explicit FaLatencyCompensationNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaLatencyCompensationNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool compensateFrame(
    const fa_interfaces::msg::AudioFrame & in,
    fa_interfaces::msg::AudioFrame & out);
  size_t bytesPerFrame() const;

  LatencyCompensationConfig config_;

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> negative_timestamp_drops_{0};
  std::atomic<uint64_t> timestamp_overflow_drops_{0};
};

}  // namespace fa_latency_compensation

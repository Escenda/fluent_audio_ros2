#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_aec_linear
{

struct AecLinearConfig
{
  bool enabled = false;

  std::string mic_topic;
  std::string ref_topic;
  std::string output_topic;

  int expected_sample_rate = -1;
  int expected_channels = -1;

  int ref_timeout_ms = -1;
  double cancel_gain = 0.0;

  int qos_depth = -1;
  bool qos_reliable = false;

  int diagnostics_publish_period_ms = -1;
};

class FaAecLinearNode : public rclcpp::Node
{
public:
  FaAecLinearNode();
  ~FaAecLinearNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void onMicFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void onRefFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg) const;
  static bool decodeToFloat(const fa_interfaces::msg::AudioFrame & msg, std::vector<float> & out_samples);
  static void encodeFromFloat(const std::vector<float> & samples, uint32_t bit_depth, std::vector<uint8_t> & out_bytes);
  static void computeRmsPeak(const std::vector<float> & interleaved, float & out_rms, float & out_peak);

  AecLinearConfig config_;

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr mic_sub_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr ref_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr out_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::mutex ref_mutex_;
  fa_interfaces::msg::AudioFrame::SharedPtr last_ref_;
  rclcpp::Time last_ref_stamp_{0, 0, RCL_ROS_TIME};

  std::atomic<uint64_t> mic_in_{0};
  std::atomic<uint64_t> mic_out_{0};
  std::atomic<uint64_t> mic_drop_{0};
  std::atomic<uint64_t> ref_in_{0};
  std::atomic<uint64_t> ref_drop_{0};
};

}  // namespace fa_aec_linear


#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_aec_nn
{

struct AecNnConfig
{
  bool enabled = false;
  std::string backend;

  std::string input_topic;
  std::string output_topic;

  int expected_sample_rate = -1;
  int expected_channels = -1;

  std::string onnx_model_path;

  int qos_depth = -1;
  bool qos_reliable = false;

  int diagnostics_publish_period_ms = -1;
};

class FaAecNnNode : public rclcpp::Node
{
public:
  FaAecNnNode();
  ~FaAecNnNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void onAudioFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg) const;

  AecNnConfig config_;

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> in_{0};
  std::atomic<uint64_t> out_{0};
  std::atomic<uint64_t> drop_{0};
};

}  // namespace fa_aec_nn


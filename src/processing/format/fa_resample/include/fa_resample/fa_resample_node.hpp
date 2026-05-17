#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_resample
{

struct ResampleConfig
{
  int target_sample_rate = -1;
  std::string input_encoding;
  int input_bit_depth = -1;
  std::string input_layout;
  std::string output_encoding;
  int output_bit_depth = -1;

  bool mic_enabled = false;
  std::string mic_input_topic;
  std::string mic_output_topic;

  bool ref_enabled = false;
  std::string ref_input_topic;
  std::string ref_output_topic;

  int qos_depth = -1;
  bool qos_reliable = false;

  int diagnostics_publish_period_ms = -1;
};

class FaResampleNode : public rclcpp::Node
{
public:
  FaResampleNode();
  ~FaResampleNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void publishDiagnostics();

  void handleMicFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void handleRefFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);

  bool processAndPublish(
    const fa_interfaces::msg::AudioFrame & in,
    const rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr & pub,
    const std::string & output_stream_id,
    std::atomic<uint64_t> & out_counter,
    std::atomic<uint64_t> & drop_counter);

  ResampleConfig config_;

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr mic_sub_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr ref_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr mic_pub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr ref_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> mic_in_{0};
  std::atomic<uint64_t> mic_out_{0};
  std::atomic<uint64_t> mic_drop_{0};
  std::atomic<uint64_t> ref_in_{0};
  std::atomic<uint64_t> ref_out_{0};
  std::atomic<uint64_t> ref_drop_{0};
};

}  // namespace fa_resample

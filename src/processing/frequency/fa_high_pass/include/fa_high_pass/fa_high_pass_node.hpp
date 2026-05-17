#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_high_pass
{

struct HighPassConfig
{
  std::string input_topic{};
  std::string output_topic{};
  double cutoff_hz{-1.0};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

struct ChannelFilterState
{
  float previous_input{0.0F};
  float previous_output{0.0F};
  bool initialized{false};
};

/**
 * @brief FLOAT32LE AudioFrame に一次 high-pass filter を適用する processing node。
 */
class FaHighPassNode : public rclcpp::Node
{
public:
  FaHighPassNode();
  ~FaHighPassNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void configureFilterState();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool applyHighPass(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  HighPassConfig config_;
  double filter_alpha_{0.0};
  std::string active_source_id_{};
  std::vector<ChannelFilterState> channel_states_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
};

}  // namespace fa_high_pass

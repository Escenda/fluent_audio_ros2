#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_hum
{

struct HumConfig
{
  std::string input_topic{};
  std::string output_topic{};
  double frequency_hz{-1.0};
  int harmonics{-1};
  double q{-1.0};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

struct BiquadCoefficients
{
  double center_hz{0.0};
  double b0{0.0};
  double b1{0.0};
  double b2{0.0};
  double a1{0.0};
  double a2{0.0};
};

struct BiquadState
{
  double previous_input_1{0.0};
  double previous_input_2{0.0};
  double previous_output_1{0.0};
  double previous_output_2{0.0};
};

using ChannelCascadeState = std::vector<BiquadState>;

/**
 * @brief FLOAT32LE interleaved AudioFrame に hum frequency と倍音の notch cascade を適用する
 * correction node。
 */
class FaHumNode : public rclcpp::Node
{
public:
  FaHumNode();
  ~FaHumNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void configureCascade();
  void resetFilterStateForSource(const std::string & source_id);

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool applyHumRemoval(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  HumConfig config_;
  std::vector<BiquadCoefficients> cascade_coefficients_{};
  std::string active_source_id_{};
  std::vector<ChannelCascadeState> channel_states_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> source_resets_{0};
};

}  // namespace fa_hum

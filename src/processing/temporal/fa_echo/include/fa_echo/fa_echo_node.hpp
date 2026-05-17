#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_echo
{

struct EchoConfig
{
  std::string input_topic{};
  std::string output_topic{};
  double delay_ms{-1.0};
  double feedback_gain{0.0};
  double wet_gain{0.0};
  double dry_gain{0.0};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

/**
 * @brief FLOAT32LE interleaved AudioFrame に内部 feedback echo を適用する processing node。
 */
class FaEchoNode : public rclcpp::Node
{
public:
  FaEchoNode();
  ~FaEchoNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool validateSamples(const fa_interfaces::msg::AudioFrame & msg);
  bool applyEcho(
    const fa_interfaces::msg::AudioFrame & in,
    fa_interfaces::msg::AudioFrame & out);

  void resetDelayState(
    std::vector<std::vector<float>> & buffers,
    std::vector<size_t> & positions) const;
  size_t bytesPerFrame() const;

  EchoConfig config_;
  size_t delay_samples_{0};
  std::string current_source_id_{};
  std::vector<std::vector<float>> delay_buffers_{};
  std::vector<size_t> delay_positions_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> messages_in_{0};
  std::atomic<uint64_t> messages_out_{0};
  std::atomic<uint64_t> messages_dropped_{0};
  std::atomic<uint64_t> source_resets_{0};
};

}  // namespace fa_echo

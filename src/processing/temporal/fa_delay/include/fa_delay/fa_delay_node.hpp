#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_delay
{

struct DelayConfig
{
  std::string input_topic{};
  std::string output_topic{};
  double delay_ms{-1.0};
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
 * @brief FLOAT32LE interleaved AudioFrame に intentional temporal delay を適用する processing node。
 */
class FaDelayNode : public rclcpp::Node
{
public:
  FaDelayNode();
  ~FaDelayNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void ensureDelayState(const std::string & source_id);
  void resetDelayState(const std::string & source_id);

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool validateSamples(const fa_interfaces::msg::AudioFrame & msg);
  bool applyDelay(
    const fa_interfaces::msg::AudioFrame & in,
    fa_interfaces::msg::AudioFrame & out);
  size_t bytesPerFrame() const;

  DelayConfig config_;
  size_t delay_samples_{0};
  std::string current_source_id_{};
  std::vector<std::deque<float>> delay_buffers_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> messages_in_{0};
  std::atomic<uint64_t> messages_out_{0};
  std::atomic<uint64_t> messages_dropped_{0};
  std::atomic<uint64_t> source_resets_{0};
};

}  // namespace fa_delay

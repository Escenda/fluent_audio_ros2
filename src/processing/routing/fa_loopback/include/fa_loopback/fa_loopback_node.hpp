#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_loopback
{

struct LoopbackConfig
{
  std::string input_topic{};
  std::string output_topic{};
  bool require_distinct_topics{true};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{true};
  int diagnostics_publish_period_ms{-1};
};

/**
 * @brief 1つの AudioFrame stream を operator-defined loopback topic へ byte-for-byte に転送する routing node。
 */
class FaLoopbackNode : public rclcpp::Node
{
public:
  FaLoopbackNode();
  ~FaLoopbackNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishLoopback(const fa_interfaces::msg::AudioFrame & msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  size_t bytesPerFrame() const;

  LoopbackConfig config_;

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
};

}  // namespace fa_loopback

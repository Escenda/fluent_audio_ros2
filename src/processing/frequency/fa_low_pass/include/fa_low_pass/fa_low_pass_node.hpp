#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_low_pass
{

namespace backends
{
class InternalFirstOrderLowPassBackend;
}  // namespace backends

struct LowPassConfig
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

/**
 * @brief FLOAT32LE AudioFrame に一次 low-pass filter を適用する processing node。
 */
class FaLowPassNode : public rclcpp::Node
{
public:
  FaLowPassNode();
  ~FaLowPassNode() override;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void configureBackend();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool applyLowPass(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  LowPassConfig config_;
  std::string active_source_id_{};
  std::unique_ptr<backends::InternalFirstOrderLowPassBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> source_resets_{0};
};

}  // namespace fa_low_pass

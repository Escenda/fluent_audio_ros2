#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_agc
{

namespace backends
{
class InternalRmsAgcBackend;
}  // namespace backends

struct AgcConfig
{
  std::string input_topic{};
  std::string output_topic{};
  double target_rms{0.1};
  double min_gain{0.25};
  double max_gain{4.0};
  double attack_ms{10.0};
  double release_ms{250.0};
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
 * @brief FLOAT32LE AudioFrame に frame RMS based automatic gain control を適用する processing node。
 */
class FaAgcNode : public rclcpp::Node
{
public:
  explicit FaAgcNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaAgcNode() override;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void configureBackend();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool applyAgc(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  AgcConfig config_;
  std::unique_ptr<backends::InternalRmsAgcBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> gain_reductions_{0};
  std::atomic<uint64_t> gain_increases_{0};
};

}  // namespace fa_agc

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_agc
{

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
  FaAgcNode();
  ~FaAgcNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool applyAgc(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);
  bool readSamples(const fa_interfaces::msg::AudioFrame & msg, std::vector<float> & samples);
  double calculateFrameRms(const std::vector<float> & samples) const;
  double boundedTargetGain(double frame_rms) const;
  double smoothingAlpha(double time_constant_ms, size_t sample_count) const;
  double smoothedGain(double target_gain, size_t sample_count) const;

  AgcConfig config_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> gain_reductions_{0};
  std::atomic<uint64_t> gain_increases_{0};
  std::atomic<double> current_gain_{1.0};
  std::atomic<double> last_frame_rms_{0.0};
  std::atomic<double> last_target_gain_{1.0};
};

}  // namespace fa_agc

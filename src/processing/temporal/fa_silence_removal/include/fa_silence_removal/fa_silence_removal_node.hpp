#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_silence_removal
{

struct SilenceRemovalConfig
{
  std::string input_topic{};
  std::string output_topic{};
  double threshold_rms{-1.0};
  double hangover_ms{-1.0};
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
 * @brief FLOAT32LE interleaved AudioFrame の silent chunk を drop する processing node。
 */
class FaSilenceRemovalNode : public rclcpp::Node
{
public:
  FaSilenceRemovalNode();
  ~FaSilenceRemovalNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishAcceptedFrame(const fa_interfaces::msg::AudioFrame & msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool computeRms(const fa_interfaces::msg::AudioFrame & msg, double & rms);
  void consumeHangoverSamples(size_t frame_count);
  size_t bytesPerFrame() const;
  size_t frameCount(const fa_interfaces::msg::AudioFrame & msg) const;

  SilenceRemovalConfig config_;
  size_t hangover_samples_{0};
  size_t hangover_samples_remaining_{0};
  double last_rms_{0.0};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> messages_in_{0};
  std::atomic<uint64_t> messages_out_{0};
  std::atomic<uint64_t> messages_dropped_{0};
  std::atomic<uint64_t> invalid_frames_dropped_{0};
  std::atomic<uint64_t> silent_frames_dropped_{0};
  std::atomic<uint64_t> active_frames_{0};
  std::atomic<uint64_t> hangover_frames_{0};
};

}  // namespace fa_silence_removal

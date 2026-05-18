#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_reverb
{

namespace backends
{
class InternalFeedbackDelayBackend;
}  // namespace backends

struct ReverbConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string input_stream_id{};
  std::string output_stream_id{};
  double room_size{-1.0};
  double damping{-1.0};
  double wet_gain{-1.0};
  double dry_gain{-1.0};
  int expected_sample_rate{0};
  int expected_channels{0};
  std::string expected_encoding{};
  int expected_bit_depth{0};
  std::string expected_layout{};
  int qos_depth{0};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{0};
  int diagnostics_qos_depth{0};
  bool diagnostics_qos_reliable{false};
};

/**
 * @brief FLOAT32LE interleaved AudioFrame に deterministic feedback-delay reverb を適用する node。
 */
class FaReverbNode : public rclcpp::Node
{
public:
  explicit FaReverbNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaReverbNode() override;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void configureBackend();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool applyReverb(
    const fa_interfaces::msg::AudioFrame & in,
    fa_interfaces::msg::AudioFrame & out);

  size_t bytesPerFrame() const;

  ReverbConfig config_;
  std::unique_ptr<backends::InternalFeedbackDelayBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> messages_in_{0};
  std::atomic<uint64_t> messages_out_{0};
  std::atomic<uint64_t> messages_dropped_{0};
  std::atomic<uint64_t> source_resets_{0};
};

}  // namespace fa_reverb

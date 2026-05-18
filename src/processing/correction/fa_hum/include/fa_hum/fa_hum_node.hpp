#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_hum
{

namespace backends
{
class InternalNotchCascadeBackend;
}  // namespace backends

struct HumConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string resolved_input_topic{};
  std::string resolved_output_topic{};
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
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

/**
 * @brief FLOAT32LE interleaved AudioFrame に hum frequency と倍音の notch cascade を適用する
 * correction node。
 */
class FaHumNode : public rclcpp::Node
{
public:
  explicit FaHumNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaHumNode() override;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void configureBackend();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool isStaleFrame(const fa_interfaces::msg::AudioFrame & msg) const;
  void rememberAcceptedFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool applyHumRemoval(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  HumConfig config_;
  std::unique_ptr<backends::InternalNotchCascadeBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> source_resets_{0};
  std::atomic<uint64_t> epoch_resets_{0};

  bool has_last_accepted_frame_{false};
  std::string last_source_id_{};
  uint32_t last_epoch_{0};
  int32_t last_stamp_sec_{0};
  uint32_t last_stamp_nanosec_{0};
};

}  // namespace fa_hum

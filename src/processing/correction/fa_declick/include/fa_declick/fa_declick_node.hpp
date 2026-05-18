#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_declick
{

namespace backends
{
class InternalImpulseDeclickBackend;
}  // namespace backends

struct DeclickConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string resolved_input_topic{};
  std::string resolved_output_topic{};
  double threshold_delta{-1.0};
  int window_max_samples{-1};
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
 * @brief FLOAT32LE interleaved AudioFrame の impulse click を明示補正する processing node。
 */
class FaDeclickNode : public rclcpp::Node
{
public:
  explicit FaDeclickNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaDeclickNode() override;

private:
  void loadParameters();
  void configureBackend();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool applyDeclick(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  DeclickConfig config_;
  std::unique_ptr<backends::InternalImpulseDeclickBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> samples_corrected_{0};
  std::atomic<uint64_t> click_runs_corrected_{0};
};

}  // namespace fa_declick

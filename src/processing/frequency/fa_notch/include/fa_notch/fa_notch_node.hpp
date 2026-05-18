#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_notch
{

namespace backends
{
class InternalNotchBackend;
}  // namespace backends

struct NotchConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string input_stream_id{};
  std::string output_stream_id{};
  double center_hz{-1.0};
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
 * @brief FLOAT32LE AudioFrame に二次 notch filter を適用する processing node。
 */
class FaNotchNode : public rclcpp::Node
{
public:
  explicit FaNotchNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaNotchNode() override;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void configureBackend();
  std::unique_ptr<backends::InternalNotchBackend> createBackend() const;

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool applyNotch(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  NotchConfig config_;
  std::string active_source_id_{};
  std::optional<uint32_t> last_epoch_{};
  std::unique_ptr<backends::InternalNotchBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> state_resets_{0};
};

}  // namespace fa_notch

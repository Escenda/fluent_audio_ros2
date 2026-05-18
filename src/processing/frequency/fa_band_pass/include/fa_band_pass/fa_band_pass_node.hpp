#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_band_pass
{

namespace backends
{
class InternalFirstOrderBandPassBackend;
}  // namespace backends

struct BandPassConfig
{
  std::string input_topic{};
  std::string output_topic{};
  double low_cut_hz{-1.0};
  double high_cut_hz{-1.0};
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
 * @brief FLOAT32LE AudioFrame に一次 high-pass 後一次 low-pass の band-pass filter を適用する node。
 */
class FaBandPassNode : public rclcpp::Node
{
public:
  explicit FaBandPassNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaBandPassNode() override;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void configureBackend();
  std::unique_ptr<backends::InternalFirstOrderBandPassBackend> createBackend() const;

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool applyBandPass(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  BandPassConfig config_;
  std::string active_source_id_{};
  std::optional<uint32_t> last_epoch_{};
  std::unique_ptr<backends::InternalFirstOrderBandPassBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> state_resets_{0};
};

}  // namespace fa_band_pass

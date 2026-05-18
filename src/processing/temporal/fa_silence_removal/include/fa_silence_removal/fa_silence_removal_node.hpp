#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_silence_removal
{

namespace backends
{
class InternalRmsSilenceRemovalBackend;
}  // namespace backends

struct SilenceRemovalConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string input_stream_id{};
  std::string output_stream_id{};
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
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

/**
 * @brief FLOAT32LE interleaved AudioFrame の silent chunk を drop する processing node。
 */
class FaSilenceRemovalNode : public rclcpp::Node
{
public:
  explicit FaSilenceRemovalNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaSilenceRemovalNode() override;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishAcceptedFrame(const fa_interfaces::msg::AudioFrame & msg);
  void publishDiagnostics();
  void configureBackend();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  size_t bytesPerFrame() const;

  SilenceRemovalConfig config_;
  size_t hangover_samples_{0};
  std::unique_ptr<backends::InternalRmsSilenceRemovalBackend> backend_{};

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

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_aec_linear
{

namespace backends
{
class BaselineLinearBackend;
}  // namespace backends

struct AecLinearConfig
{
  bool enabled = false;

  std::string mic_topic;
  std::string ref_topic;
  std::string output_topic;
  std::string resolved_mic_topic;
  std::string resolved_ref_topic;
  std::string resolved_output_topic;

  int expected_sample_rate = -1;
  int expected_channels = -1;
  std::string expected_encoding;
  int expected_bit_depth = -1;

  int ref_timeout_ms = -1;
  std::string reference_failure_policy = "drop";
  double cancel_gain = 0.0;

  int qos_depth = -1;
  bool qos_reliable = false;

  int diagnostics_publish_period_ms = -1;
};

enum class FrameValidationStatus
{
  kOk,
  kMissingSourceId,
  kStreamIdMismatch,
  kInvalidTimestamp,
  kSampleRateMismatch,
  kChannelsMismatch,
  kFormatMismatch,
  kLayoutMismatch,
  kEmptyData,
  kMisalignedData,
};

class FaAecLinearNode : public rclcpp::Node
{
public:
  explicit FaAecLinearNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaAecLinearNode() override;

private:
  void loadParameters();
  void configureBackend();
  void setupInterfaces();
  void onMicFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void onRefFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  [[nodiscard]] FrameValidationStatus validateFrame(
    const fa_interfaces::msg::AudioFrame & msg,
    const std::string & expected_stream_id) const;

  AecLinearConfig config_;
  std::unique_ptr<backends::BaselineLinearBackend> backend_;

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr mic_sub_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr ref_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr out_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::mutex ref_mutex_;
  fa_interfaces::msg::AudioFrame::SharedPtr last_ref_;
  rclcpp::Time last_ref_stamp_{0, 0, RCL_ROS_TIME};

  std::atomic<uint64_t> mic_in_{0};
  std::atomic<uint64_t> mic_out_{0};
  std::atomic<uint64_t> mic_drop_{0};
  std::atomic<uint64_t> ref_in_{0};
  std::atomic<uint64_t> ref_drop_{0};
};

}  // namespace fa_aec_linear

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "builtin_interfaces/msg/time.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_clock_drift
{

struct ClockDriftConfig
{
  std::string input_topic{};
  std::string output_topic{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  double drift_ema_alpha{-1.0};
  double drift_max_correction_ms_per_frame{-1.0};
  double drift_reset_threshold_ms{-1.0};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

struct ActiveStreamIdentity
{
  std::string source_id{};
  std::string stream_id{};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  std::string encoding{};
  uint32_t bit_depth{0};
  std::string layout{};
};

/**
 * @brief AudioFrame の payload を変えず、sample-clock drift による timestamp metadata だけを補正する node。
 */
class FaClockDriftNode : public rclcpp::Node
{
public:
  explicit FaClockDriftNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaClockDriftNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool correctFrame(
    const fa_interfaces::msg::AudioFrame & in,
    fa_interfaces::msg::AudioFrame & out);
  bool publishBaselineFrame(
    const fa_interfaces::msg::AudioFrame & in,
    fa_interfaces::msg::AudioFrame & out,
    long double last_observed_drift_ns,
    long double current_frame_duration_ns);
  bool hasDifferentStreamIdentity(const fa_interfaces::msg::AudioFrame & msg) const;
  void activateStream(const fa_interfaces::msg::AudioFrame & msg, long double output_timestamp_ns);
  void resetTimeline();
  bool frameDurationNanoseconds(const fa_interfaces::msg::AudioFrame & msg, long double & duration_ns) const;
  bool stampToNanoseconds(
    const builtin_interfaces::msg::Time & stamp,
    long double & timestamp_ns) const;
  bool buildStamp(long double timestamp_ns, builtin_interfaces::msg::Time & stamp) const;
  long double boundCorrectionNanoseconds(
    long double correction_ns,
    long double previous_frame_duration_ns) const;
  size_t bytesPerSampleFrame() const;

  ClockDriftConfig config_;

  mutable std::mutex timeline_mutex_;
  std::optional<ActiveStreamIdentity> active_stream_{};
  std::optional<long double> previous_output_timestamp_ns_{};
  std::optional<long double> previous_frame_duration_ns_{};
  long double drift_estimate_ns_{0.0L};
  long double last_observed_drift_ns_{0.0L};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> timeline_resets_{0};
  std::atomic<uint64_t> correction_limited_frames_{0};
};

}  // namespace fa_clock_drift

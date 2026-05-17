#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_ducking
{

struct DuckingConfig
{
  std::string program_topic{};
  std::string sidechain_topic{};
  std::string output_topic{};
  double sidechain_threshold_rms{-1.0};
  int sidechain_max_age_ms{-1};
  double ducking_gain_db{0.0};
  double ducking_gain_linear{1.0};
  double attack_ms{-1.0};
  double release_ms{-1.0};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

struct SidechainSnapshot
{
  bool available{false};
  double rms{0.0};
  rclcpp::Time received_at{0, 0, RCL_ROS_TIME};
};

/**
 * @brief Program AudioFrame に sidechain RMS based ducking を適用する routing/dynamics node。
 */
class FaDuckingNode : public rclcpp::Node
{
public:
  FaDuckingNode();
  ~FaDuckingNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleProgramFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void handleSidechainFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(
    const fa_interfaces::msg::AudioFrame & msg,
    const std::string & expected_stream_id,
    const char * input_name);
  bool readSamples(
    const fa_interfaces::msg::AudioFrame & msg,
    std::vector<float> & samples,
    const char * input_name);
  double calculateFrameRms(const std::vector<float> & samples) const;
  void invalidateSidechainState();
  SidechainSnapshot sidechainSnapshot() const;
  bool sidechainIsActive(const rclcpp::Time & now);
  double smoothingAlpha(double time_constant_ms, size_t sample_count) const;
  double smoothedGain(double target_gain, size_t sample_count) const;
  bool applyDucking(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  DuckingConfig config_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr program_sub_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr sidechain_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  mutable std::mutex sidechain_mutex_;
  bool has_sidechain_{false};
  double latest_sidechain_rms_{0.0};
  rclcpp::Time latest_sidechain_received_at_{0, 0, RCL_ROS_TIME};

  std::atomic<uint64_t> program_frames_in_{0};
  std::atomic<uint64_t> program_frames_out_{0};
  std::atomic<uint64_t> program_frames_dropped_{0};
  std::atomic<uint64_t> sidechain_frames_in_{0};
  std::atomic<uint64_t> sidechain_frames_valid_{0};
  std::atomic<uint64_t> sidechain_frames_dropped_{0};
  std::atomic<uint64_t> sidechain_state_invalidations_{0};
  std::atomic<uint64_t> ducked_program_frames_{0};
  std::atomic<uint64_t> released_program_frames_{0};
  std::atomic<uint64_t> stale_sidechain_checks_{0};
  std::atomic<double> current_gain_{1.0};
  std::atomic<double> last_target_gain_{1.0};
  std::atomic<double> last_sidechain_rms_{0.0};
  std::atomic<int64_t> last_sidechain_age_ms_{-1};
  std::atomic<bool> last_sidechain_active_{false};
};

}  // namespace fa_ducking

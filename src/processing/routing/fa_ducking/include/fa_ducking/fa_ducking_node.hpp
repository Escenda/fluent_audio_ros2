#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_ducking
{

namespace backends
{
class InternalSidechainDuckingBackend;
}  // namespace backends

struct DuckingConfig
{
  std::string program_topic{};
  std::string sidechain_topic{};
  std::string output_topic{};
  std::string program_stream_id{};
  std::string sidechain_stream_id{};
  std::string output_stream_id{};
  double sidechain_threshold_rms{-1.0};
  int sidechain_max_age_ms{-1};
  double ducking_gain_db{0.0};
  double attack_ms{-1.0};
  double release_ms{-1.0};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

/**
 * @brief Program AudioFrame に sidechain RMS based ducking を適用する routing/dynamics node。
 */
class FaDuckingNode : public rclcpp::Node
{
public:
  explicit FaDuckingNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaDuckingNode() override;

private:
  void loadParameters();
  void configureBackend();
  void setupInterfaces();
  void handleProgramFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void handleSidechainFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(
    const fa_interfaces::msg::AudioFrame & msg,
    const std::string & expected_stream_id,
    const char * input_name);
  size_t bytesPerFrame() const;
  int64_t nowNanoseconds() const;

  DuckingConfig config_;
  std::unique_ptr<backends::InternalSidechainDuckingBackend> backend_{};
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr program_sub_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr sidechain_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

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
};

}  // namespace fa_ducking

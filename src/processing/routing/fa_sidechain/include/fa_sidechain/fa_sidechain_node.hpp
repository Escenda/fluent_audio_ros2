#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_sidechain
{

struct SidechainConfig
{
  std::string sidechain_topic{};
  std::string control_topic{};
  double threshold_rms{-1.0};
  double active_gain_db{std::numeric_limits<double>::quiet_NaN()};
  double inactive_gain_db{std::numeric_limits<double>::quiet_NaN()};
  double active_gain_linear{1.0};
  double inactive_gain_linear{1.0};
  int control_sample_rate{-1};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

/**
 * @brief Sidechain AudioFrame を明示的な mono gain-control AudioFrame へ変換する routing node。
 *
 * @details 入力波形は解析だけに使い、program audio は購読も変更もしない。
 */
class FaSidechainNode : public rclcpp::Node
{
public:
  FaSidechainNode();
  ~FaSidechainNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleSidechainFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool readSamples(const fa_interfaces::msg::AudioFrame & msg, std::vector<float> & samples);
  double calculateFrameRms(const std::vector<float> & samples) const;
  double targetGainForRms(double rms) const;
  bool buildControlFrame(
    const fa_interfaces::msg::AudioFrame & input,
    double rms,
    fa_interfaces::msg::AudioFrame & output);

  SidechainConfig config_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr sidechain_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr control_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> active_frames_{0};
  std::atomic<uint64_t> inactive_frames_{0};
  std::atomic<double> last_rms_{0.0};
  std::atomic<double> last_gain_linear_{1.0};
};

}  // namespace fa_sidechain

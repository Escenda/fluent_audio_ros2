#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_channel_convert
{

struct ChannelConvertConfig
{
  std::string input_topic{};
  std::string output_topic{};
  int input_channels{-1};
  int output_channels{-1};
  std::string conversion_mode{};
  int expected_sample_rate{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

/**
 * @brief FLOAT32LE interleaved AudioFrame の channel count だけを明示的に変換する processing node。
 *
 * @details device I/O、resample、sample format 変換、gain、limiter、noise gate、filtering、denoise は扱わない。
 * 起動時に mode と channel 契約を検証し、runtime frame が契約と一致しない場合は publish せず drop する。
 */
class FaChannelConvertNode : public rclcpp::Node
{
public:
  FaChannelConvertNode();
  ~FaChannelConvertNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool convertFrame(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  static bool isSupportedConversion(const std::string & mode, int input_channels, int output_channels);
  static float readFloat32Le(const std::vector<uint8_t> & bytes, size_t sample_index);
  static void appendFloat32Le(float sample, std::vector<uint8_t> & out_bytes);
  static bool isNormalizedFinite(float sample);

  ChannelConvertConfig config_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
};

}  // namespace fa_channel_convert

#pragma once

#include <atomic>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_channel_convert/backends/internal_float32le_channel_convert.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_channel_convert
{

struct ChannelConvertConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string input_stream_id{};
  std::string output_stream_id{};
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
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
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

  ChannelConvertConfig config_;
  std::unique_ptr<backends::InternalFloat32LeChannelConvertBackend> backend_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
};

}  // namespace fa_channel_convert

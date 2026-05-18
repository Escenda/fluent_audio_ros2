#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_bit_depth/backends/internal_integer_bit_depth.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_bit_depth
{

struct BitDepthConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string input_stream_id{};
  std::string output_stream_id{};
  std::string input_encoding{};
  int input_bit_depth{-1};
  std::string output_encoding{};
  int output_bit_depth{-1};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

/**
 * @brief AudioFrame の PCM integer bit depth だけを明示的に変換する processing node。
 *
 * @details sample format normalization、device I/O、codec、resample、gain、channel 変換は扱わない。
 * 起動時に変換契約を検証し、runtime frame が契約と一致しない場合は publish せず drop する。
 */
class FaBitDepthNode : public rclcpp::Node
{
public:
  explicit FaBitDepthNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaBitDepthNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool convertFrame(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  BitDepthConfig config_;
  std::unique_ptr<backends::InternalIntegerBitDepthBackend> backend_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
};

}  // namespace fa_bit_depth

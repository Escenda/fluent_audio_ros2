#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interleave/backends/internal_layout_reorder.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_interleave
{

struct InterleaveConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string input_layout{};
  std::string output_layout{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

/**
 * @brief AudioFrame の interleaved / planar layout だけを明示的に変換する processing node。
 *
 * @details device I/O、codec decode、resample、sample format conversion、bit-depth conversion、
 * channel count conversion、gain、filtering は扱わない。起動時に layout/format 契約を検証し、
 * runtime frame が契約と一致しない場合は warning を出して drop する。
 */
class FaInterleaveNode : public rclcpp::Node
{
public:
  FaInterleaveNode();
  ~FaInterleaveNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool convertFrame(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  InterleaveConfig config_;
  std::unique_ptr<backends::InternalLayoutReorderBackend> backend_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
};

}  // namespace fa_interleave

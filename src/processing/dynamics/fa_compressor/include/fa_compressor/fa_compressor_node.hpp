#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_compressor
{

namespace backends
{
class InternalStaticCurveBackend;
}  // namespace backends

struct CompressorConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string input_stream_id{};
  std::string output_stream_id{};
  double threshold_linear;
  double ratio;
  double makeup_gain_linear;
  int expected_sample_rate;
  int expected_channels;
  std::string expected_encoding{};
  int expected_bit_depth;
  std::string expected_layout{};
  int qos_depth;
  bool qos_reliable;
  int diagnostics_qos_depth;
  bool diagnostics_qos_reliable;
  int diagnostics_publish_period_ms;
};

/**
 * @brief FLOAT32LE AudioFrame に static per-sample compressor curve を適用する processing node。
 */
class FaCompressorNode : public rclcpp::Node
{
public:
  explicit FaCompressorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaCompressorNode() override;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void configureBackend();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool applyCompressor(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  CompressorConfig config_;
  std::unique_ptr<backends::InternalStaticCurveBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> samples_compressed_{0};
};

}  // namespace fa_compressor

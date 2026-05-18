#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_denoise
{

struct DenoiseConfig
{
  bool enabled = false;
  std::string backend_name;

  std::string input_topic;
  std::string output_topic;
  std::string resolved_input_topic;
  std::string resolved_output_topic;

  int expected_sample_rate = -1;
  int expected_channels = -1;
  std::string expected_encoding;
  int expected_bit_depth = -1;

  std::string output_encoding;
  int output_bit_depth = -1;

  // DTLN (ONNX) backend parameters
  int dtln_block_len = -1;
  int dtln_block_shift = -1;
  std::string dtln_model_1_path;
  std::string dtln_model_2_path;
  int dtln_intra_op_num_threads = -1;
  int dtln_inter_op_num_threads = -1;
  bool dtln_enable_ort_optimizations = true;

  int qos_depth = -1;
  bool qos_reliable = false;

  int diagnostics_publish_period_ms = -1;
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

namespace backends
{
class DenoiseBackend;
}

class FaDenoiseNode : public rclcpp::Node
{
public:
  explicit FaDenoiseNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaDenoiseNode() override;

private:
  void loadParameters();
  void configureBackend();
  void setupInterfaces();
  void onAudioFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg) const;

  DenoiseConfig config_;
  std::unique_ptr<backends::DenoiseBackend> backend_;

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> in_{0};
  std::atomic<uint64_t> out_{0};
  std::atomic<uint64_t> drop_{0};

  std::atomic<uint64_t> process_ns_sum_{0};
  std::atomic<uint64_t> process_ns_max_{0};
  std::atomic<uint64_t> process_count_{0};

};

}  // namespace fa_denoise

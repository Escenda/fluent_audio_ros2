#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_trim
{

namespace backends
{
class InternalFrameTrimBackend;
}  // namespace backends

struct TrimConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string input_stream_id{};
  std::string output_stream_id{};
  int leading_frames{-1};
  int trailing_frames{-1};
  int expected_sample_rate{-1};
  int expected_channels{-1};
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
 * @brief FLOAT32LE interleaved AudioFrame の先頭・末尾 sample frame を削除する temporal node。
 */
class FaTrimNode : public rclcpp::Node
{
public:
  explicit FaTrimNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaTrimNode() override;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void configureBackend();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool applyTrim(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);
  size_t bytesPerFrame() const;

  TrimConfig config_;
  std::unique_ptr<backends::InternalFrameTrimBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> contract_drops_{0};
  std::atomic<uint64_t> invalid_sample_drops_{0};
  std::atomic<uint64_t> trim_exhausted_drops_{0};
  std::atomic<uint64_t> epoch_overflow_drops_{0};
  std::atomic<uint64_t> last_input_frame_count_{0};
  std::atomic<uint64_t> last_output_frame_count_{0};
};

}  // namespace fa_trim

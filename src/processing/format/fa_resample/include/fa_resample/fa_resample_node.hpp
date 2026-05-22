#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_resample
{

namespace backends
{
class ResamplerBackend;
}  // namespace backends

struct ResampleConfig
{
  int target_sample_rate = -1;
  std::string backend_name;
  int backend_speex_quality = -1;
  std::string backend_soxr_quality;
  std::string backend_quality_label;

  std::string input_encoding;
  int input_bit_depth = -1;
  std::string input_layout;
  std::string output_encoding;
  int output_bit_depth = -1;

  bool mic_enabled = false;
  std::string mic_input_topic;
  std::string mic_output_topic;
  std::string mic_input_stream_id;
  std::string mic_output_stream_id;

  bool ref_enabled = false;
  std::string ref_input_topic;
  std::string ref_output_topic;
  std::string ref_input_stream_id;
  std::string ref_output_stream_id;

  int qos_depth = -1;
  bool qos_reliable = false;

  int diagnostics_publish_period_ms = -1;
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

class FaResampleNode : public rclcpp::Node
{
public:
  explicit FaResampleNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaResampleNode() override;

private:
  struct OutputTimelineState
  {
    bool started = false;
    rclcpp::Time base_stamp{0, 0, RCL_ROS_TIME};
    uint64_t output_frames_published = 0;
  };

  void loadParameters();
  void setupInterfaces();
  void publishDiagnostics();
  void configureBackend();

  void handleMicFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void handleRefFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);

  bool processAndPublish(
    const fa_interfaces::msg::AudioFrame & in,
    const rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr & pub,
    const std::string & expected_input_stream_id,
    const std::string & output_stream_id,
    OutputTimelineState & output_timeline,
    std::atomic<uint64_t> & out_counter,
    std::atomic<uint64_t> & drop_counter);

  rclcpp::Time outputFrameStamp(
    const builtin_interfaces::msg::Time & input_stamp,
    OutputTimelineState & output_timeline) const;
  static rclcpp::Duration mediaOffsetFromFrames(
    uint64_t frame_count,
    int sample_rate);

  ResampleConfig config_;
  std::unique_ptr<backends::ResamplerBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr mic_sub_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr ref_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr mic_pub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr ref_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> mic_in_{0};
  std::atomic<uint64_t> mic_out_{0};
  std::atomic<uint64_t> mic_drop_{0};
  std::atomic<uint64_t> ref_in_{0};
  std::atomic<uint64_t> ref_out_{0};
  std::atomic<uint64_t> ref_drop_{0};

  OutputTimelineState mic_output_timeline_{};
  OutputTimelineState ref_output_timeline_{};
};

}  // namespace fa_resample

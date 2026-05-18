#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_encode/backends/external_codec_encoder.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_interfaces/msg/encoded_audio_chunk.hpp"

namespace fa_encode
{

struct FaEncodeConfig
{
  std::string backend_name{};
  std::string command_executable{};
  std::vector<std::string> command_arguments{};
  int command_timeout_ms{-1};
  int command_max_output_bytes{-1};
  std::string input_topic{};
  std::string output_topic{};
  std::string input_stream_id{};
  std::string output_stream_id{};
  int input_sample_rate{-1};
  int input_channels{-1};
  std::string input_encoding{};
  int input_bit_depth{-1};
  std::string input_layout{};
  std::string output_codec{};
  std::string output_container{};
  std::string output_payload_format{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

/**
 * @brief PCM AudioFrame を明示 codec backend で EncodedAudioChunk へ変換する node。
 *
 * @details device I/O、resample、bit-depth conversion、gain、VAD/ASR は扱わない。
 * backend は ROS2 message を知らず、node が AudioFrame と EncodedAudioChunk の契約を検証する。
 */
class FaEncodeNode : public rclcpp::Node
{
public:
  explicit FaEncodeNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaEncodeNode() override = default;

private:
  void loadParameters();
  void setupBackend();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool buildChunk(
    const fa_interfaces::msg::AudioFrame & in,
    const backends::EncodeResult & result,
    fa_interfaces::msg::EncodedAudioChunk & out);
  uint64_t durationNsFromFrame(const fa_interfaces::msg::AudioFrame & msg) const;
  size_t bytesPerFrame() const;

  FaEncodeConfig config_;
  std::unique_ptr<backends::ExternalCodecEncoderBackend> backend_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::EncodedAudioChunk>::SharedPtr encoded_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  bool have_epoch_{false};
  uint32_t active_epoch_{0};
  uint64_t next_sequence_{0};
  uint64_t next_media_time_ns_{0};

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> chunks_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> encoded_bytes_out_{0};
};

}  // namespace fa_encode
